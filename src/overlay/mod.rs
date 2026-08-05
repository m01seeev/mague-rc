use std::{
    cell::{Cell, RefCell},
    fs, io,
    path::PathBuf,
    process,
    rc::Rc,
    sync::mpsc,
    thread,
};

use futures_util::StreamExt;
use gtk::{
    Align, Application, ApplicationWindow, Box as GtkBox, Button, Image, Label, Orientation,
    PolicyType, ScrolledWindow, Separator, Stack, TextBuffer, TextView, gdk, gio, glib, prelude::*,
};
use gtk4_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};
use thiserror::Error;
use tokio::sync::mpsc as tokio_mpsc;

use crate::{
    app,
    config::Config,
    events::{
        AppCommand, AppErrorView, LiveCodingState, Mode, OutputComponent, OutputEvent, StatusKind,
        StatusMessage,
    },
    output::{
        AnswerStatus, AppSnapshot, ChannelOutputSink, ConnectionStatus, ConversationTurn,
        WorkerStatus,
    },
    telemetry::TelemetryOutputSink,
};

mod style;

use style::install_css;

const APPLICATION_ID: &str = "io.github.mague_rc.Overlay";
const NAMESPACE: &str = "mague-rc-overlay";
const WIDTH: i32 = 1120;
const HEIGHT: i32 = 900;
const COLLAPSED_WIDTH: i32 = 420;
const COLLAPSED_HEIGHT: i32 = 52;
const CONTROL_PID_FILE: &str = "/tmp/mague-rc-overlay.pid";

#[derive(Debug, Error)]
pub enum OverlayError {
    #[error("could not write overlay control PID file `{path}`: {source}")]
    ControlPidFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("GTK application exited with status {0}")]
    Exit(i32),
}

pub fn run(config: Config) -> Result<(), OverlayError> {
    let _pid_file = OverlayPidFile::create()?;
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

struct OverlayPidFile {
    path: PathBuf,
    pid: u32,
}

impl OverlayPidFile {
    fn create() -> Result<Self, OverlayError> {
        let path = control_pid_path();
        let pid = process::id();
        fs::write(&path, pid.to_string()).map_err(|source| OverlayError::ControlPidFile {
            path: path.clone(),
            source,
        })?;
        Ok(Self { path, pid })
    }
}

impl Drop for OverlayPidFile {
    fn drop(&mut self) {
        let owns_file = fs::read_to_string(&self.path)
            .is_ok_and(|contents| contents.trim() == self.pid.to_string());
        if owns_file {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn control_pid_path() -> PathBuf {
    PathBuf::from(CONTROL_PID_FILE)
}

struct OverlayState {
    snapshot: AppSnapshot,
    live_coding: LiveCodingState,
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
            live_coding: LiveCodingState::default(),
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
            OutputEvent::ModeChanged { mode } => {
                self.status = match mode {
                    Mode::LiveCoding => String::from("Live coding"),
                    Mode::Voice | Mode::Ocr => String::from("Listening"),
                };
            }
            OutputEvent::Status(status) => {
                self.status = status_label(status.kind).to_owned();
                match status.kind {
                    StatusKind::Paused => self.paused = true,
                    StatusKind::Listening => self.paused = false,
                    StatusKind::HistoryCleared => {
                        self.live_coding = LiveCodingState::default();
                    }
                    _ => {}
                }
            }
            OutputEvent::Error(error) if error.component != OutputComponent::Knowledge => {
                self.status = String::from("Error");
            }
            OutputEvent::AnswerStarted(meta) => {
                self.status = if meta.mode == Mode::LiveCoding {
                    if meta.speaker == crate::events::Speaker::Candidate {
                        String::from("Coaching")
                    } else {
                        String::from("Updating code")
                    }
                } else if meta.speaker == crate::events::Speaker::Candidate {
                    String::from("Coaching")
                } else {
                    String::from("Thinking")
                };
            }
            OutputEvent::AnswerCompleted { .. } if !self.paused => {
                self.status = String::from("Listening");
            }
            OutputEvent::LiveCodingUpdated(updated) => {
                self.live_coding.clone_from(updated);
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
    mode_stack: Stack,
    interview_tab: Button,
    coding_tab: Button,
    live_coding: LiveCodingWidgets,
    footer: Label,
    queue: Label,
    collapse_button: Button,
    collapse_icon: Image,
    body: GtkBox,
}

#[derive(Clone)]
struct LiveCodingWidgets {
    revision: Label,
    explanation: Label,
    summary: Label,
    language: Label,
    change_note: Label,
    code: TextView,
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
                Ok(runtime) if config.session_log.enabled => {
                    let sink = TelemetryOutputSink::new_session(
                        ChannelOutputSink::new(output_sender),
                        &config.session_log.directory,
                        "training",
                        &config,
                    )
                    .map_err(|error| crate::error::AppError::Output(error.to_string()));
                    match sink {
                        Ok(sink) => runtime.block_on(app::run_with_sink(
                            config,
                            sink,
                            command_receiver,
                        )),
                        Err(error) => Err(error),
                    }
                }
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
    root.set_cursor_from_name(Some("default"));
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

    let mode_tabs = GtkBox::new(Orientation::Horizontal, 6);
    mode_tabs.add_css_class("mode-tabs");
    mode_tabs.set_margin_top(8);
    mode_tabs.set_margin_start(12);
    mode_tabs.set_margin_end(12);
    mode_tabs.set_margin_bottom(8);
    let interview_tab = mode_button("INTERVIEW", "Use interview Q&A mode");
    let coding_tab = mode_button("LIVE CODING", "Use stateful live-coding mode · F10");
    mode_tabs.append(&interview_tab);
    mode_tabs.append(&coding_tab);
    body.append(&mode_tabs);

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

    let coding_page = GtkBox::new(Orientation::Vertical, 0);
    coding_page.add_css_class("coding-page");

    let coding_meta = GtkBox::new(Orientation::Horizontal, 8);
    coding_meta.set_margin_start(16);
    coding_meta.set_margin_end(12);
    coding_meta.set_margin_bottom(8);
    let revision = Label::new(Some("STATE r0"));
    revision.add_css_class("coding-revision");
    revision.set_halign(Align::Start);
    coding_meta.append(&revision);
    let coding_meta_spacer = GtkBox::new(Orientation::Horizontal, 0);
    coding_meta_spacer.set_hexpand(true);
    coding_meta.append(&coding_meta_spacer);
    let language = Label::new(Some("JAVA"));
    language.add_css_class("coding-language");
    coding_meta.append(&language);
    let copy_button = icon_button("edit-copy-symbolic", "Copy stable code");
    {
        let state = Rc::clone(&state);
        copy_button.connect_clicked(move |_| {
            let code = state.borrow().live_coding.code.clone();
            if !code.is_empty()
                && let Some(display) = gdk::Display::default()
            {
                display.clipboard().set_text(&code);
            }
        });
    }
    coding_meta.append(&copy_button);
    coding_page.append(&coding_meta);

    let coding_workspace = GtkBox::new(Orientation::Horizontal, 0);
    coding_workspace.add_css_class("coding-workspace");
    coding_workspace.set_vexpand(true);

    let context_column = GtkBox::new(Orientation::Vertical, 10);
    context_column.add_css_class("coding-context");
    context_column.set_width_request(380);
    context_column.set_margin_start(16);
    context_column.set_margin_end(14);
    context_column.set_margin_bottom(14);

    let explanation_root = GtkBox::new(Orientation::Vertical, 5);
    explanation_root.add_css_class("coding-state-block");
    explanation_root.add_css_class("coding-talk-block");
    explanation_root.set_vexpand(true);
    explanation_root.append(&section_title("TALK TRACK", true));
    let explanation = content_label(
        "A short explanation will appear with the solution",
        "coding-explanation",
    );
    let explanation_scroll = ScrolledWindow::builder()
        .hscrollbar_policy(PolicyType::Never)
        .vscrollbar_policy(PolicyType::Automatic)
        .vexpand(true)
        .child(&explanation)
        .build();
    explanation_scroll.add_css_class("coding-text-scroll");
    explanation_root.append(&explanation_scroll);
    context_column.append(&explanation_root);

    let summary_root = GtkBox::new(Orientation::Vertical, 5);
    summary_root.add_css_class("coding-state-block");
    summary_root.set_vexpand(true);
    summary_root.append(&section_title("CANONICAL STATE", false));
    let summary = content_label("Summary will appear after the first task segment", "coding-summary");
    let summary_scroll = ScrolledWindow::builder()
        .hscrollbar_policy(PolicyType::Never)
        .vscrollbar_policy(PolicyType::Automatic)
        .vexpand(true)
        .child(&summary)
        .build();
    summary_scroll.add_css_class("coding-text-scroll");
    summary_root.append(&summary_scroll);
    context_column.append(&summary_root);
    coding_workspace.append(&context_column);
    coding_workspace.append(&Separator::new(Orientation::Vertical));

    let code_column = GtkBox::new(Orientation::Vertical, 0);
    code_column.add_css_class("coding-editor");
    code_column.set_hexpand(true);
    let code_header = GtkBox::new(Orientation::Horizontal, 8);
    code_header.set_margin_start(14);
    code_header.set_margin_end(16);
    code_header.set_margin_bottom(6);
    code_header.append(&section_title("STABLE CODE", true));
    let code_header_spacer = GtkBox::new(Orientation::Horizontal, 0);
    code_header_spacer.set_hexpand(true);
    code_header.append(&code_header_spacer);
    let change_note = Label::new(Some("No generated code yet"));
    change_note.add_css_class("coding-change-note");
    change_note.set_ellipsize(gtk::pango::EllipsizeMode::End);
    code_header.append(&change_note);
    code_column.append(&code_header);

    let code_buffer = TextBuffer::new(None);
    let _ = code_buffer.create_tag(
        Some("changed-line"),
        &[("paragraph-background", &"#17343a")],
    );
    let code = TextView::builder()
        .buffer(&code_buffer)
        .editable(false)
        .cursor_visible(false)
        .monospace(true)
        .wrap_mode(gtk::WrapMode::None)
        .left_margin(14)
        .right_margin(14)
        .top_margin(12)
        .bottom_margin(12)
        .build();
    code.add_css_class("coding-code");
    code.set_cursor_from_name(Some("default"));
    let code_scroll = ScrolledWindow::builder()
        .hscrollbar_policy(PolicyType::Automatic)
        .vscrollbar_policy(PolicyType::Automatic)
        .vexpand(true)
        .child(&code)
        .build();
    code_scroll.add_css_class("coding-code-scroll");
    code_column.append(&code_scroll);
    coding_workspace.append(&code_column);
    coding_page.append(&coding_workspace);

    let mode_stack = Stack::new();
    mode_stack.set_vexpand(true);
    mode_stack.add_named(&history_scroll, Some("interview"));
    mode_stack.add_named(&coding_page, Some("live-coding"));
    mode_stack.set_visible_child_name("interview");
    body.append(&mode_stack);

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
        mode_stack,
        interview_tab: interview_tab.clone(),
        coding_tab: coding_tab.clone(),
        live_coding: LiveCodingWidgets {
            revision,
            explanation,
            summary,
            language,
            change_note,
            code,
        },
        footer,
        queue,
        collapse_button: collapse_button.clone(),
        collapse_icon,
        body: body.clone(),
    };
    force_default_cursor(widgets.window.upcast_ref());

    {
        let state = Rc::clone(&state);
        interview_tab.connect_clicked(move |_| {
            let mut state = state.borrow_mut();
            if state.snapshot.mode == Mode::LiveCoding {
                state.send(AppCommand::ToggleLiveCoding);
            }
        });
    }
    {
        let state = Rc::clone(&state);
        coding_tab.connect_clicked(move |_| {
            let mut state = state.borrow_mut();
            if state.snapshot.mode != Mode::LiveCoding {
                state.send(AppCommand::ToggleLiveCoding);
            }
        });
    }

    {
        let state = Rc::clone(&state);
        let widgets = widgets.clone();
        collapse_button.connect_clicked(move |_| {
            toggle_collapsed(&state, &widgets);
        });
    }
    listen_for_control_signals(Rc::clone(&state), widgets.clone());

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

#[derive(Clone, Copy)]
enum OverlayControl {
    ToggleCollapsed,
    ToggleLiveCoding,
}

fn listen_for_control_signals(state: Rc<RefCell<OverlayState>>, widgets: OverlayWidgets) {
    let (signal_sender, mut signal_receiver) = futures_channel::mpsc::unbounded();
    let spawn_result = thread::Builder::new()
        .name(String::from("mague-rc-overlay-control"))
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            let Ok(runtime) = runtime else {
                tracing::error!("could not create overlay control runtime");
                return;
            };
            runtime.block_on(async move {
                let collapse_signal = tokio::signal::unix::signal(
                    tokio::signal::unix::SignalKind::user_defined2(),
                );
                let mode_signal = tokio::signal::unix::signal(
                    tokio::signal::unix::SignalKind::window_change(),
                );
                let (Ok(mut collapse_signal), Ok(mut mode_signal)) =
                    (collapse_signal, mode_signal)
                else {
                    tracing::error!("could not install overlay control signal handlers");
                    return;
                };
                loop {
                    let control = tokio::select! {
                        signal = collapse_signal.recv() => {
                            signal.map(|_| OverlayControl::ToggleCollapsed)
                        }
                        signal = mode_signal.recv() => {
                            signal.map(|_| OverlayControl::ToggleLiveCoding)
                        }
                    };
                    let Some(control) = control else {
                        return;
                    };
                    if signal_sender.unbounded_send(control).is_err() {
                        return;
                    }
                }
            });
        });

    if let Err(error) = spawn_result {
        state
            .borrow_mut()
            .apply_output(&OutputEvent::Error(AppErrorView {
                component: OutputComponent::App,
                message: format!("could not start overlay control: {error}"),
            }));
        refresh_widgets(&state.borrow(), &widgets);
        return;
    }

    glib::spawn_future_local(async move {
        while let Some(control) = signal_receiver.next().await {
            match control {
                OverlayControl::ToggleCollapsed => toggle_collapsed(&state, &widgets),
                OverlayControl::ToggleLiveCoding => {
                    state.borrow_mut().send(AppCommand::ToggleLiveCoding);
                }
            }
        }
    });
}

fn toggle_collapsed(state: &Rc<RefCell<OverlayState>>, widgets: &OverlayWidgets) {
    let collapsed = {
        let mut state = state.borrow_mut();
        state.collapsed = !state.collapsed;
        state.collapsed
    };
    widgets.body.set_visible(!collapsed);
    if collapsed {
        widgets
            .window
            .set_default_size(COLLAPSED_WIDTH, COLLAPSED_HEIGHT);
        widgets
            .collapse_icon
            .set_icon_name(Some("go-down-symbolic"));
        widgets
            .collapse_button
            .set_tooltip_text(Some("Expand"));
    } else {
        widgets.window.set_default_size(WIDTH, HEIGHT);
        widgets.body.set_visible(true);
        widgets
            .collapse_icon
            .set_icon_name(Some("go-up-symbolic"));
        widgets
            .collapse_button
            .set_tooltip_text(Some("Collapse"));
    }
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
    button.set_cursor_from_name(Some("default"));
    button
}

fn mode_button(label: &str, tooltip: &str) -> Button {
    let button = Button::with_label(label);
    button.add_css_class("mode-tab");
    button.set_tooltip_text(Some(tooltip));
    button.set_cursor_from_name(Some("default"));
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
    label.set_selectable(false);
    label.set_cursor_from_name(Some("default"));
    label.add_css_class(css_class);
    label
}

fn force_default_cursor(widget: &gtk::Widget) {
    widget.set_cursor_from_name(Some("default"));
    let mut child = widget.first_child();
    while let Some(current) = child {
        child = current.next_sibling();
        force_default_cursor(&current);
    }
}

fn create_conversation_turn(turn: &ConversationTurn) -> ConversationTurnWidgets {
    let root = GtkBox::new(Orientation::Vertical, 0);
    root.add_css_class("conversation-turn");

    let question_section = GtkBox::new(Orientation::Vertical, 6);
    question_section.set_margin_top(14);
    question_section.set_margin_bottom(12);
    question_section.set_margin_start(16);
    question_section.set_margin_end(16);
    let (question_title, answer_title) = if turn.speaker == crate::events::Speaker::Candidate {
        ("YOU", "COACH")
    } else {
        ("QUESTION", "ANSWER")
    };
    question_section.append(&section_title(question_title, true));

    let question = content_label(&turn.question, "question");
    question_section.append(&question);
    root.append(&question_section);

    let answer_section = GtkBox::new(Orientation::Vertical, 8);
    answer_section.set_margin_bottom(16);
    answer_section.set_margin_start(16);
    answer_section.set_margin_end(16);
    answer_section.append(&section_title(answer_title, false));

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

    let live_coding = state.snapshot.mode == Mode::LiveCoding;
    widgets
        .mode_stack
        .set_visible_child_name(if live_coding {
            "live-coding"
        } else {
            "interview"
        });
    widgets.interview_tab.remove_css_class("active");
    widgets.coding_tab.remove_css_class("active");
    if live_coding {
        widgets.coding_tab.add_css_class("active");
    } else {
        widgets.interview_tab.add_css_class("active");
    }

    refresh_conversation_history(&state.snapshot.conversation, &widgets.history);
    refresh_transcript_draft(&state.snapshot.transcript_draft, &widgets.history);
    refresh_live_coding(state, &widgets.live_coding);

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

fn refresh_live_coding(state: &OverlayState, widgets: &LiveCodingWidgets) {
    widgets
        .revision
        .set_text(&format!("STATE r{}", state.live_coding.revision));
    widgets.explanation.set_text(
        (!state.live_coding.explanation.is_empty())
            .then_some(state.live_coding.explanation.as_str())
            .unwrap_or("A short explanation will appear with the solution"),
    );
    widgets.summary.set_text(
        (!state.live_coding.summary.is_empty())
            .then_some(state.live_coding.summary.as_str())
            .unwrap_or("Summary will appear after the first task segment"),
    );
    widgets.language.set_text(
        (!state.live_coding.language.is_empty())
            .then_some(state.live_coding.language.as_str())
            .unwrap_or("JAVA"),
    );
    widgets.change_note.set_text(
        (!state.live_coding.change_note.is_empty())
            .then_some(state.live_coding.change_note.as_str())
            .unwrap_or("No generated code yet"),
    );

    let buffer = widgets.code.buffer();
    let (start, end) = buffer.bounds();
    let current_code = buffer.text(&start, &end, true);
    if current_code.as_str() == state.live_coding.code {
        return;
    }
    apply_code_edits(&buffer, &state.live_coding.code_edits);
    let (start, end) = buffer.bounds();
    if buffer.text(&start, &end, true).as_str() != state.live_coding.code {
        buffer.set_text(&state.live_coding.code);
    }
    let (start, end) = buffer.bounds();
    buffer.remove_all_tags(&start, &end);
    for line in &state.live_coding.changed_lines {
        let Ok(line_index) = i32::try_from(line.saturating_sub(1)) else {
            continue;
        };
        let Some(start) = buffer.iter_at_line(line_index) else {
            continue;
        };
        let mut end = start;
        if !end.forward_line() {
            end = buffer.end_iter();
        }
        buffer.apply_tag_by_name("changed-line", &start, &end);
    }
}

fn apply_code_edits(buffer: &TextBuffer, edits: &[crate::events::CodeEdit]) {
    for edit in edits.iter().rev() {
        let (Ok(start_offset), Ok(end_offset)) = (
            i32::try_from(edit.start_offset),
            i32::try_from(edit.end_offset),
        ) else {
            return;
        };
        let mut start = buffer.iter_at_offset(start_offset);
        let mut end = buffer.iter_at_offset(end_offset);
        buffer.delete(&mut start, &mut end);
        buffer.insert(&mut start, &edit.replacement);
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
        WorkerStatus::Working if state.snapshot.mode == Mode::LiveCoding => {
            String::from("Updating canonical state and code")
        }
        WorkerStatus::Working => String::from("Generating answer"),
        WorkerStatus::Error => String::from("Answer provider error"),
        WorkerStatus::Idle if state.snapshot.mode == Mode::LiveCoding => {
            String::from("Live coding · interviewer + candidate mic · RAG off")
        }
        WorkerStatus::Idle if state.snapshot.listening => {
            String::from("Interviewer + candidate mic")
        }
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
            mode: Mode::Voice,
            speaker: crate::events::Speaker::Interviewer,
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
            speaker: crate::events::Speaker::Interviewer,
            text: "Как работает".to_owned(),
        });
        state.apply_output(&OutputEvent::TranscriptDraft {
            speaker: crate::events::Speaker::Interviewer,
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
