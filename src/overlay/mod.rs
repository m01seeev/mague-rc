use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::mpsc,
    thread,
};

use futures_util::StreamExt;
use gtk::{
    Align, Application, ApplicationWindow, Box as GtkBox, Button, CssProvider, Image, Label,
    Orientation, PolicyType, STYLE_PROVIDER_PRIORITY_APPLICATION, ScrolledWindow, Separator, gdk,
    gio, glib, prelude::*,
};
use gtk4_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};
use thiserror::Error;
use tokio::sync::mpsc as tokio_mpsc;

use crate::{
    app,
    config::Config,
    events::{AppCommand, AppErrorView, OutputComponent, OutputEvent, StatusKind, StatusMessage},
    output::{
        AnswerStatus, AppSnapshot, ChannelOutputSink, ConnectionStatus, ConversationTurn,
        WorkerStatus,
    },
};

const APPLICATION_ID: &str = "io.github.mague_rc.Overlay";
const NAMESPACE: &str = "mague-rc-overlay";
const WIDTH: i32 = 760;
const HEIGHT: i32 = 600;
const COLLAPSED_WIDTH: i32 = 420;
const COLLAPSED_HEIGHT: i32 = 52;

#[derive(Debug, Error)]
pub enum OverlayError {
    #[error("GTK application exited with status {0}")]
    Exit(i32),
}

pub fn run(config: Config) -> Result<(), OverlayError> {
    let application = Application::new(Some(APPLICATION_ID), gio::ApplicationFlags::NON_UNIQUE);

    application.connect_startup(|_| install_css());
    application.connect_activate(move |application| {
        build_overlay(application, config.clone());
    });

    let exit_code = i32::from(application.run());
    if exit_code == 0 {
        Ok(())
    } else {
        Err(OverlayError::Exit(exit_code))
    }
}

struct OverlayState {
    snapshot: AppSnapshot,
    status: String,
    paused: bool,
    collapsed: bool,
    stopping: bool,
    commands: tokio_mpsc::UnboundedSender<AppCommand>,
}

impl OverlayState {
    fn new(commands: tokio_mpsc::UnboundedSender<AppCommand>) -> Self {
        Self {
            snapshot: AppSnapshot::default(),
            status: String::from("Starting"),
            paused: false,
            collapsed: false,
            stopping: false,
            commands,
        }
    }

    fn send(&mut self, command: AppCommand) {
        if self.commands.send(command).is_err() {
            self.snapshot.apply(&OutputEvent::Error(AppErrorView {
                component: OutputComponent::App,
                message: String::from("pipeline command channel closed"),
            }));
            self.status = String::from("Error");
        }
    }

    fn apply_output(&mut self, event: &OutputEvent) -> bool {
        let should_exit = matches!(
            event,
            OutputEvent::Status(StatusMessage {
                kind: StatusKind::Stopped,
                ..
            })
        );

        match event {
            OutputEvent::Status(status) => {
                self.status = status_label(status.kind).to_owned();
                match status.kind {
                    StatusKind::Paused => self.paused = true,
                    StatusKind::Listening => self.paused = false,
                    _ => {}
                }
            }
            OutputEvent::Error(_) => self.status = String::from("Error"),
            OutputEvent::AnswerStarted(_) => self.status = String::from("Thinking"),
            OutputEvent::AnswerCompleted { .. } if !self.paused => {
                self.status = String::from("Listening");
            }
            _ => {}
        }

        self.snapshot.apply(event);
        should_exit
    }
}

impl Drop for OverlayState {
    fn drop(&mut self) {
        let _ = self.commands.send(AppCommand::Shutdown);
    }
}

#[derive(Clone)]
struct OverlayWidgets {
    window: ApplicationWindow,
    status_dot: GtkBox,
    status_label: Label,
    pause_icon: Image,
    history: ConversationHistoryWidgets,
    footer: Label,
    queue: Label,
    collapse_icon: Image,
    body: GtkBox,
}

#[derive(Clone)]
struct ConversationHistoryWidgets {
    turns_container: GtkBox,
    empty: Label,
    draft_root: GtkBox,
    draft: Label,
    turns: Rc<RefCell<Vec<ConversationTurnWidgets>>>,
}

struct ConversationTurnWidgets {
    request_id: u64,
    root: GtkBox,
    question: Label,
    answer: Label,
}

fn build_overlay(application: &Application, config: Config) {
    let (output_sender, output_receiver) = mpsc::channel();
    let (command_sender, command_receiver) = tokio_mpsc::unbounded_channel();
    let state = Rc::new(RefCell::new(OverlayState::new(command_sender)));

    let widgets = create_widgets(application, Rc::clone(&state));
    widgets.window.present();

    spawn_pipeline(config, output_sender, command_receiver);
    bridge_output_events(
        application.clone(),
        output_receiver,
        Rc::clone(&state),
        widgets,
    );
}

fn spawn_pipeline(
    config: Config,
    output_sender: mpsc::Sender<OutputEvent>,
    command_receiver: tokio_mpsc::UnboundedReceiver<AppCommand>,
) {
    let error_sender = output_sender.clone();
    let thread_error_sender = error_sender.clone();
    let spawn_result = thread::Builder::new()
        .name(String::from("mague-rc-pipeline"))
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build();

            let result = match runtime {
                Ok(runtime) => runtime.block_on(app::run_with_sink(
                    config,
                    ChannelOutputSink::new(output_sender),
                    command_receiver,
                )),
                Err(error) => Err(crate::error::AppError::Output(format!(
                    "failed to create pipeline runtime: {error}"
                ))),
            };

            if let Err(error) = result {
                let _ = thread_error_sender.send(OutputEvent::Error(AppErrorView {
                    component: OutputComponent::App,
                    message: error.to_string(),
                }));
                let _ = thread_error_sender.send(OutputEvent::Status(StatusMessage {
                    kind: StatusKind::Stopped,
                    text: String::from("pipeline stopped"),
                }));
            }
        });

    if let Err(error) = spawn_result {
        let _ = error_sender.send(OutputEvent::Error(AppErrorView {
            component: OutputComponent::App,
            message: format!("failed to start pipeline thread: {error}"),
        }));
        let _ = error_sender.send(OutputEvent::Status(StatusMessage {
            kind: StatusKind::Stopped,
            text: String::from("pipeline stopped"),
        }));
    }
}

fn bridge_output_events(
    application: Application,
    output_receiver: mpsc::Receiver<OutputEvent>,
    state: Rc<RefCell<OverlayState>>,
    widgets: OverlayWidgets,
) {
    let (ui_sender, mut ui_receiver) = futures_channel::mpsc::unbounded();

    let bridge_result = thread::Builder::new()
        .name(String::from("mague-rc-ui-events"))
        .spawn(move || {
            while let Ok(event) = output_receiver.recv() {
                if ui_sender.unbounded_send(event).is_err() {
                    break;
                }
            }
        });

    if let Err(error) = bridge_result {
        state
            .borrow_mut()
            .apply_output(&OutputEvent::Error(AppErrorView {
                component: OutputComponent::App,
                message: format!("failed to start UI event bridge: {error}"),
            }));
        refresh_widgets(&state.borrow(), &widgets);
        return;
    }

    glib::spawn_future_local(async move {
        while let Some(event) = ui_receiver.next().await {
            let should_exit = state.borrow_mut().apply_output(&event);
            refresh_widgets(&state.borrow(), &widgets);

            if should_exit {
                application.quit();
                break;
            }
        }
    });
}

fn create_widgets(application: &Application, state: Rc<RefCell<OverlayState>>) -> OverlayWidgets {
    let window = ApplicationWindow::builder()
        .application(application)
        .default_width(WIDTH)
        .default_height(HEIGHT)
        .decorated(false)
        .resizable(false)
        .build();
    window.add_css_class("overlay-window");
    window.init_layer_shell();
    window.set_namespace(Some(NAMESPACE));
    window.set_layer(Layer::Overlay);
    window.set_keyboard_mode(KeyboardMode::None);
    window.set_anchor(Edge::Top, true);
    window.set_margin(Edge::Top, 24);
    window.set_exclusive_zone(-1);

    let root = GtkBox::new(Orientation::Vertical, 0);
    root.add_css_class("panel");
    window.set_child(Some(&root));

    let header = GtkBox::new(Orientation::Horizontal, 8);
    header.set_height_request(51);
    header.set_margin_start(16);
    header.set_margin_end(10);
    header.set_valign(Align::Center);

    let brand = Label::new(Some("mague-rc"));
    brand.add_css_class("brand");
    brand.set_halign(Align::Start);
    header.append(&brand);

    let header_spacer = GtkBox::new(Orientation::Horizontal, 0);
    header_spacer.set_hexpand(true);
    header.append(&header_spacer);

    let status_dot = GtkBox::new(Orientation::Horizontal, 0);
    status_dot.add_css_class("status-dot");
    status_dot.add_css_class("status-muted");
    status_dot.set_size_request(8, 8);
    status_dot.set_halign(Align::Center);
    status_dot.set_valign(Align::Center);
    header.append(&status_dot);

    let status_label = Label::new(Some("Starting"));
    status_label.add_css_class("status-label");
    header.append(&status_label);

    let pause_button = icon_button("media-playback-pause-symbolic", "Pause");
    let pause_icon = pause_button
        .child()
        .and_downcast::<Image>()
        .expect("pause button must contain an image");
    {
        let state = Rc::clone(&state);
        let pause_icon = pause_icon.clone();
        pause_button.connect_clicked(move |_| {
            let mut state = state.borrow_mut();
            if state.paused {
                state.send(AppCommand::ResumeListening);
                pause_icon.set_icon_name(Some("media-playback-pause-symbolic"));
            } else {
                state.send(AppCommand::PauseListening);
                pause_icon.set_icon_name(Some("media-playback-start-symbolic"));
            }
        });
    }
    header.append(&pause_button);

    let clear_button = icon_button("user-trash-symbolic", "Clear history");
    {
        let state = Rc::clone(&state);
        clear_button.connect_clicked(move |_| {
            state.borrow_mut().send(AppCommand::ClearHistory);
        });
    }
    header.append(&clear_button);

    let collapse_button = icon_button("go-up-symbolic", "Collapse");
    let collapse_icon = collapse_button
        .child()
        .and_downcast::<Image>()
        .expect("collapse button must contain an image");
    header.append(&collapse_button);

    let close_button = icon_button("window-close-symbolic", "Close");
    {
        let state = Rc::clone(&state);
        let status_label = status_label.clone();
        close_button.connect_clicked(move |_| {
            let mut state = state.borrow_mut();
            if !state.stopping {
                state.stopping = true;
                state.status = String::from("Stopping");
                status_label.set_text("Stopping");
                state.send(AppCommand::Shutdown);
            }
        });
    }
    header.append(&close_button);
    root.append(&header);
    root.append(&Separator::new(Orientation::Horizontal));

    let body = GtkBox::new(Orientation::Vertical, 0);
    body.set_vexpand(true);
    root.append(&body);

    let history_container = GtkBox::new(Orientation::Vertical, 0);
    let history_empty = Label::new(Some("Listening for a question"));
    history_empty.add_css_class("empty-history");
    history_empty.set_margin_top(32);
    history_empty.set_margin_bottom(32);
    history_container.append(&history_empty);

    let turns_container = GtkBox::new(Orientation::Vertical, 0);
    history_container.append(&turns_container);

    let draft_root = GtkBox::new(Orientation::Vertical, 6);
    draft_root.add_css_class("transcript-draft");
    draft_root.set_margin_top(14);
    draft_root.set_margin_bottom(16);
    draft_root.set_margin_start(16);
    draft_root.set_margin_end(16);
    draft_root.append(&section_title("QUESTION / LIVE", true));
    let draft = content_label("", "question");
    draft_root.append(&draft);
    draft_root.set_visible(false);
    history_container.append(&draft_root);

    let history_scroll = ScrolledWindow::builder()
        .hscrollbar_policy(PolicyType::Never)
        .vscrollbar_policy(PolicyType::Automatic)
        .vexpand(true)
        .child(&history_container)
        .build();
    configure_follow_tail(&history_scroll);
    body.append(&history_scroll);

    let footer_row = GtkBox::new(Orientation::Horizontal, 8);
    footer_row.add_css_class("footer");
    footer_row.set_margin_top(10);
    footer_row.set_margin_start(16);
    footer_row.set_margin_end(16);
    footer_row.set_margin_bottom(13);

    let footer = Label::new(Some("Connecting to speech recognition"));
    footer.set_halign(Align::Start);
    footer.set_ellipsize(gtk::pango::EllipsizeMode::End);
    footer.set_hexpand(true);
    footer_row.append(&footer);

    let queue = Label::new(None);
    queue.set_halign(Align::End);
    queue.set_visible(false);
    footer_row.append(&queue);
    body.append(&footer_row);

    let widgets = OverlayWidgets {
        window: window.clone(),
        status_dot,
        status_label: status_label.clone(),
        pause_icon,
        history: ConversationHistoryWidgets {
            turns_container,
            empty: history_empty,
            draft_root,
            draft,
            turns: Rc::new(RefCell::new(Vec::new())),
        },
        footer,
        queue,
        collapse_icon,
        body: body.clone(),
    };

    {
        let state = Rc::clone(&state);
        let widgets = widgets.clone();
        collapse_button.connect_clicked(move |button| {
            let mut state = state.borrow_mut();
            state.collapsed = !state.collapsed;
            if state.collapsed {
                widgets.body.set_visible(false);
                widgets
                    .window
                    .set_default_size(COLLAPSED_WIDTH, COLLAPSED_HEIGHT);
                widgets
                    .collapse_icon
                    .set_icon_name(Some("go-down-symbolic"));
                button.set_tooltip_text(Some("Expand"));
            } else {
                widgets.window.set_default_size(WIDTH, HEIGHT);
                widgets.body.set_visible(true);
                widgets.collapse_icon.set_icon_name(Some("go-up-symbolic"));
                button.set_tooltip_text(Some("Collapse"));
            }
        });
    }

    let state_for_close = Rc::clone(&state);
    let status_for_close = status_label.clone();
    window.connect_close_request(move |_| {
        let mut state = state_for_close.borrow_mut();
        if !state.stopping {
            state.stopping = true;
            state.status = String::from("Stopping");
            status_for_close.set_text("Stopping");
            state.send(AppCommand::Shutdown);
        }
        glib::Propagation::Stop
    });

    widgets
}

fn icon_button(icon_name: &str, tooltip: &str) -> Button {
    let image = Image::from_icon_name(icon_name);
    image.set_pixel_size(17);
    let button = Button::builder()
        .child(&image)
        .tooltip_text(tooltip)
        .width_request(34)
        .height_request(34)
        .build();
    button.add_css_class("icon-button");
    button
}

fn section_title(value: &str, accent: bool) -> Label {
    let label = Label::new(Some(value));
    label.set_halign(Align::Start);
    label.add_css_class("section-title");
    if accent {
        label.add_css_class("accent");
    }
    label
}

fn content_label(value: &str, css_class: &str) -> Label {
    let label = Label::new(Some(value));
    label.set_halign(Align::Start);
    label.set_valign(Align::Start);
    label.set_wrap(true);
    label.set_wrap_mode(gtk::pango::WrapMode::WordChar);
    label.set_xalign(0.0);
    label.set_selectable(true);
    label.add_css_class(css_class);
    label
}

fn create_conversation_turn(turn: &ConversationTurn) -> ConversationTurnWidgets {
    let root = GtkBox::new(Orientation::Vertical, 0);
    root.add_css_class("conversation-turn");

    let question_section = GtkBox::new(Orientation::Vertical, 6);
    question_section.set_margin_top(14);
    question_section.set_margin_bottom(12);
    question_section.set_margin_start(16);
    question_section.set_margin_end(16);
    question_section.append(&section_title("QUESTION", true));

    let question = content_label(&turn.question, "question");
    question_section.append(&question);
    root.append(&question_section);

    let answer_section = GtkBox::new(Orientation::Vertical, 8);
    answer_section.set_margin_bottom(16);
    answer_section.set_margin_start(16);
    answer_section.set_margin_end(16);
    answer_section.append(&section_title("ANSWER", false));

    let answer = content_label("", "answer");
    update_answer(&answer, turn);
    answer_section.append(&answer);
    root.append(&answer_section);
    root.append(&Separator::new(Orientation::Horizontal));

    ConversationTurnWidgets {
        request_id: turn.request_id,
        root,
        question,
        answer,
    }
}

fn refresh_conversation_history(
    conversation: &[ConversationTurn],
    history: &ConversationHistoryWidgets,
) {
    let mut turns = history.turns.borrow_mut();

    let needs_rebuild = turns.len() > conversation.len()
        || turns
            .iter()
            .zip(conversation)
            .any(|(widgets, turn)| widgets.request_id != turn.request_id);
    if needs_rebuild {
        for turn in turns.drain(..) {
            history.turns_container.remove(&turn.root);
        }
    }

    for turn in conversation.iter().skip(turns.len()) {
        let widgets = create_conversation_turn(turn);
        history.turns_container.append(&widgets.root);
        turns.push(widgets);
    }

    for (widgets, turn) in turns.iter().zip(conversation) {
        if widgets.question.text() != turn.question {
            widgets.question.set_text(&turn.question);
        }
        update_answer(&widgets.answer, turn);
    }

    history
        .empty
        .set_visible(conversation.is_empty() && history.draft.text().is_empty());
}

fn refresh_transcript_draft(text: &str, history: &ConversationHistoryWidgets) {
    history.draft.set_text(text);
    history.draft_root.set_visible(!text.is_empty());
    history
        .empty
        .set_visible(history.turns.borrow().is_empty() && text.is_empty());
}

fn configure_follow_tail(scroll: &ScrolledWindow) {
    let adjustment = scroll.vadjustment();
    let follow_tail = Rc::new(Cell::new(true));

    adjustment.connect_value_changed({
        let follow_tail = Rc::clone(&follow_tail);
        move |adjustment| follow_tail.set(adjustment_is_at_bottom(adjustment))
    });
    adjustment.connect_changed(move |adjustment| {
        if follow_tail.get() {
            scroll_adjustment_to_bottom(adjustment);
        }
    });
}

fn adjustment_is_at_bottom(adjustment: &gtk::Adjustment) -> bool {
    scroll_position_is_at_bottom(
        adjustment.value(),
        adjustment.page_size(),
        adjustment.upper(),
    )
}

fn scroll_position_is_at_bottom(value: f64, page_size: f64, upper: f64) -> bool {
    value + page_size >= upper - 12.0
}

fn scroll_adjustment_to_bottom(adjustment: &gtk::Adjustment) {
    let target = (adjustment.upper() - adjustment.page_size()).max(adjustment.lower());
    adjustment.set_value(target);
}

fn update_answer(label: &Label, turn: &ConversationTurn) -> bool {
    let text = if turn.answer.is_empty() {
        match turn.answer_status {
            AnswerStatus::Pending => "Waiting for answer",
            AnswerStatus::Streaming => "Thinking...",
            AnswerStatus::Completed => "No answer returned",
            AnswerStatus::Failed => "Answer failed",
        }
    } else {
        &turn.answer
    };
    let changed = label.text() != text;
    if changed {
        label.set_text(text);
    }

    label.remove_css_class("answer-placeholder");
    label.remove_css_class("answer-error");
    if turn.answer.is_empty() {
        match turn.answer_status {
            AnswerStatus::Failed => label.add_css_class("answer-error"),
            AnswerStatus::Pending | AnswerStatus::Streaming | AnswerStatus::Completed => {
                label.add_css_class("answer-placeholder");
            }
        }
    }

    changed
}

fn refresh_widgets(state: &OverlayState, widgets: &OverlayWidgets) {
    widgets.status_label.set_text(&state.status);
    widgets.pause_icon.set_icon_name(Some(if state.paused {
        "media-playback-start-symbolic"
    } else {
        "media-playback-pause-symbolic"
    }));

    for class in [
        "status-muted",
        "status-listening",
        "status-working",
        "status-error",
    ] {
        widgets.status_dot.remove_css_class(class);
    }
    widgets.status_dot.add_css_class(status_css_class(state));

    refresh_conversation_history(&state.snapshot.conversation, &widgets.history);
    refresh_transcript_draft(&state.snapshot.transcript_draft, &widgets.history);

    widgets.footer.set_text(&status_detail(state));
    let has_queue = state.snapshot.audio_queue_len > 0 || state.snapshot.llm_queue_len > 0;
    widgets.queue.set_visible(has_queue);
    if has_queue {
        widgets.queue.set_text(&format!(
            "audio {}  |  answers {}",
            state.snapshot.audio_queue_len, state.snapshot.llm_queue_len
        ));
    }
}

fn status_css_class(state: &OverlayState) -> &'static str {
    if state.snapshot.last_error.is_some() {
        return "status-error";
    }
    if state.paused
        || matches!(
            state.snapshot.stt_status,
            ConnectionStatus::Connecting | ConnectionStatus::Reconnecting
        )
        || state.snapshot.llm_status == WorkerStatus::Working
    {
        return "status-working";
    }
    if state.snapshot.listening {
        "status-listening"
    } else {
        "status-muted"
    }
}

fn status_detail(state: &OverlayState) -> String {
    if let Some(error) = &state.snapshot.last_error {
        return format!("{}: {}", error.component, error.message);
    }
    if state.paused {
        return String::from("Audio transcription paused");
    }
    match state.snapshot.llm_status {
        WorkerStatus::Working => String::from("Generating answer"),
        WorkerStatus::Error => String::from("Answer provider error"),
        WorkerStatus::Idle if state.snapshot.listening => String::from("System audio"),
        WorkerStatus::Idle => String::from("Connecting to speech recognition"),
    }
}

fn status_label(kind: StatusKind) -> &'static str {
    match kind {
        StatusKind::Started => "Starting",
        StatusKind::Connecting => "Connecting",
        StatusKind::Listening => "Listening",
        StatusKind::Paused => "Paused",
        StatusKind::Reconnecting => "Reconnecting",
        StatusKind::HistoryCleared => "History cleared",
        StatusKind::Stopped => "Stopped",
    }
}

fn install_css() {
    let provider = CssProvider::new();
    provider.load_from_string(
        r#"
        window.overlay-window {
            background: transparent;
        }

        .panel {
            background: rgba(14, 15, 17, 0.90);
            color: #edf0f2;
            border: 1px solid #383d45;
            border-radius: 8px;
        }

        .brand {
            color: #edf0f2;
            font-family: "JetBrains Mono", monospace;
            font-size: 18px;
            font-weight: 600;
        }

        .status-label,
        .footer {
            color: #969da6;
            font-family: "Noto Sans", sans-serif;
            font-size: 12px;
        }

        .status-dot {
            min-width: 8px;
            min-height: 8px;
            border-radius: 4px;
        }

        .status-muted { background: #969da6; }
        .status-listening { background: #45c778; }
        .status-working { background: #f5ad40; }
        .status-error { background: #ef595c; }

        .icon-button {
            min-width: 34px;
            min-height: 34px;
            padding: 0;
            color: #edf0f2;
            background: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            box-shadow: none;
        }

        .icon-button:hover {
            background: #181a1d;
            border-color: #383d45;
        }

        .icon-button:active {
            background: #25292e;
        }

        separator {
            min-height: 1px;
            background: #383d45;
        }

        .section-title {
            color: #969da6;
            font-family: "JetBrains Mono", monospace;
            font-size: 11px;
            font-weight: 600;
        }

        .section-title.accent {
            color: #52bacf;
        }

        .question {
            color: #edf0f2;
            font-family: "Noto Sans", sans-serif;
            font-size: 15px;
        }

        .answer {
            color: #edf0f2;
            font-family: "Noto Sans", sans-serif;
            font-size: 16px;
        }

        .answer-placeholder,
        .empty-history {
            color: #777f89;
        }

        .answer-error {
            color: #ef595c;
        }

        .empty-history {
            font-family: "Noto Sans", sans-serif;
            font-size: 14px;
        }

        scrolledwindow,
        scrolledwindow viewport {
            background: transparent;
            border: 0;
        }

        scrollbar slider {
            min-width: 6px;
            min-height: 24px;
            background: #4b525c;
            border-radius: 3px;
        }
        "#,
    );

    if let Some(display) = gdk::Display::default() {
        gtk::style_context_add_provider_for_display(
            &display,
            &provider,
            STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> OverlayState {
        let (command_sender, _command_receiver) = tokio_mpsc::unbounded_channel();
        OverlayState::new(command_sender)
    }

    #[test]
    fn applies_streaming_output_to_overlay_state() {
        let mut state = state();
        state.apply_output(&OutputEvent::Status(StatusMessage {
            kind: StatusKind::Listening,
            text: String::from("connected"),
        }));
        state.apply_output(&OutputEvent::Transcript(crate::events::TranscriptView {
            sequence: 1,
            text: String::from("What is ownership?"),
            flush_reason: String::from("test"),
        }));

        assert_eq!(state.status, "Listening");
        assert!(state.snapshot.listening);
        assert_eq!(state.snapshot.current_transcript, "What is ownership?");
        assert_eq!(state.snapshot.conversation.len(), 1);
        assert_eq!(
            state.snapshot.conversation[0].question,
            "What is ownership?"
        );
    }

    #[test]
    fn stopped_status_requests_ui_exit() {
        let mut state = state();
        assert!(state.apply_output(&OutputEvent::Status(StatusMessage {
            kind: StatusKind::Stopped,
            text: String::from("stopped"),
        })));
    }

    #[test]
    fn applies_live_transcript_draft_to_overlay_state() {
        let mut state = state();
        state.apply_output(&OutputEvent::TranscriptDraft {
            text: "Как работает".to_owned(),
        });
        state.apply_output(&OutputEvent::TranscriptDraft {
            text: "Как работает HashMap?".to_owned(),
        });

        assert_eq!(state.snapshot.transcript_draft, "Как работает HashMap?");
    }

    #[test]
    fn detects_when_scroll_position_is_at_the_bottom() {
        assert!(scroll_position_is_at_bottom(80.0, 20.0, 100.0));
        assert!(scroll_position_is_at_bottom(69.0, 20.0, 100.0));
        assert!(!scroll_position_is_at_bottom(60.0, 20.0, 100.0));
    }
}
