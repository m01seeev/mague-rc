use std::{sync::mpsc, thread, time::Duration};

use iced::{
    Alignment, Background, Border, Color, Element, Length, Subscription, Task, Theme, border,
    widget::{Space, button, column, container, row, rule, scrollable, text, tooltip},
};
use iced_layershell::{
    application, disable_clipboard,
    reexport::{Anchor, KeyboardInteractivity, Layer},
    settings::{LayerShellSettings, Settings, StartMode},
    to_layer_message,
};
use lucide_icons::{
    LUCIDE_FONT_BYTES,
    iced::{icon_chevron_down, icon_chevron_up, icon_pause, icon_play, icon_trash_2, icon_x},
};
use thiserror::Error;
use tokio::sync::mpsc as tokio_mpsc;

use crate::{
    app,
    config::Config,
    events::{AppCommand, AppErrorView, OutputComponent, OutputEvent, StatusKind, StatusMessage},
    output::{AppSnapshot, ChannelOutputSink, ConnectionStatus, WorkerStatus},
};

const WIDTH: u32 = 760;
const HEIGHT: u32 = 420;
const COLLAPSED_WIDTH: u32 = 230;
const COLLAPSED_HEIGHT: u32 = 52;
const POLL_INTERVAL: Duration = Duration::from_millis(50);

const TRANSPARENT: Color = Color::TRANSPARENT;
const PANEL: Color = Color::from_rgba(0.055, 0.059, 0.067, 0.96);
const SURFACE: Color = Color::from_rgba(0.094, 0.102, 0.114, 0.98);
const BORDER: Color = Color::from_rgb(0.22, 0.24, 0.27);
const TEXT: Color = Color::from_rgb(0.93, 0.94, 0.95);
const MUTED: Color = Color::from_rgb(0.59, 0.62, 0.66);
const GREEN: Color = Color::from_rgb(0.27, 0.78, 0.47);
const AMBER: Color = Color::from_rgb(0.96, 0.68, 0.25);
const RED: Color = Color::from_rgb(0.94, 0.35, 0.36);
const CYAN: Color = Color::from_rgb(0.32, 0.73, 0.82);

#[derive(Debug, Error)]
pub enum OverlayError {
    #[error("failed to run Wayland overlay: {0}")]
    LayerShell(#[from] iced_layershell::Error),
}

pub fn run(config: Config) -> Result<(), OverlayError> {
    disable_clipboard();

    application(
        move || OverlayState::new(config.clone()),
        || String::from("mague-rc-overlay"),
        update,
        view,
    )
    .subscription(subscription)
    .theme(app_theme)
    .style(app_style)
    .settings(Settings {
        id: Some(String::from("mague-rc-overlay")),
        fonts: vec![LUCIDE_FONT_BYTES.into()],
        antialiasing: true,
        layer_settings: LayerShellSettings {
            anchor: Anchor::Top | Anchor::Right,
            layer: Layer::Overlay,
            exclusive_zone: -1,
            size: Some((WIDTH, HEIGHT)),
            margin: (24, 24, 0, 0),
            keyboard_interactivity: KeyboardInteractivity::None,
            start_mode: StartMode::Active,
            events_transparent: false,
        },
        ..Settings::default()
    })
    .run()
    .map_err(OverlayError::from)
}

struct OverlayState {
    snapshot: AppSnapshot,
    status: String,
    paused: bool,
    collapsed: bool,
    stopping: bool,
    events: mpsc::Receiver<OutputEvent>,
    commands: tokio_mpsc::UnboundedSender<AppCommand>,
}

impl OverlayState {
    fn new(config: Config) -> Self {
        let (output_sender, output_receiver) = mpsc::channel();
        let (command_sender, command_receiver) = tokio_mpsc::unbounded_channel();
        let error_sender = output_sender.clone();

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
                    let _ = error_sender.send(OutputEvent::Error(AppErrorView {
                        component: OutputComponent::App,
                        message: error.to_string(),
                    }));
                    let _ = error_sender.send(OutputEvent::Status(StatusMessage {
                        kind: StatusKind::Stopped,
                        text: String::from("pipeline stopped"),
                    }));
                }
            });

        let mut state = Self {
            snapshot: AppSnapshot::default(),
            status: String::from("Starting"),
            paused: false,
            collapsed: false,
            stopping: false,
            events: output_receiver,
            commands: command_sender,
        };

        if let Err(error) = spawn_result {
            state.snapshot.apply(&OutputEvent::Error(AppErrorView {
                component: OutputComponent::App,
                message: format!("failed to start pipeline thread: {error}"),
            }));
            state.status = String::from("Error");
        }

        state
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

    fn apply_output(&mut self, event: OutputEvent) -> bool {
        let should_exit = matches!(
            event,
            OutputEvent::Status(StatusMessage {
                kind: StatusKind::Stopped,
                ..
            })
        );

        match &event {
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

        self.snapshot.apply(&event);
        should_exit
    }
}

impl Drop for OverlayState {
    fn drop(&mut self) {
        let _ = self.commands.send(AppCommand::Shutdown);
    }
}

#[to_layer_message]
#[derive(Clone, Debug)]
enum Message {
    Tick,
    TogglePause,
    ClearHistory,
    ToggleCollapsed,
    Close,
}

fn subscription(_state: &OverlayState) -> Subscription<Message> {
    iced::time::every(POLL_INTERVAL).map(|_| Message::Tick)
}

fn update(state: &mut OverlayState, message: Message) -> Task<Message> {
    match message {
        Message::Tick => {
            let events: Vec<_> = state.events.try_iter().collect();
            if events.into_iter().any(|event| state.apply_output(event)) {
                iced::exit()
            } else {
                Task::none()
            }
        }
        Message::TogglePause => {
            if state.paused {
                state.send(AppCommand::ResumeListening);
            } else {
                state.send(AppCommand::PauseListening);
            }
            Task::none()
        }
        Message::ClearHistory => {
            state.send(AppCommand::ClearHistory);
            Task::none()
        }
        Message::ToggleCollapsed => {
            state.collapsed = !state.collapsed;
            let size = if state.collapsed {
                (COLLAPSED_WIDTH, COLLAPSED_HEIGHT)
            } else {
                (WIDTH, HEIGHT)
            };
            Task::done(Message::SizeChange(size))
        }
        Message::Close => {
            if !state.stopping {
                state.stopping = true;
                state.status = String::from("Stopping");
                state.send(AppCommand::Shutdown);
            }
            Task::none()
        }
        _ => Task::none(),
    }
}

fn view(state: &OverlayState) -> Element<'_, Message> {
    let status_color = status_color(state);
    let controls = row![
        icon_button(
            if state.paused {
                icon_play().size(17).color(TEXT)
            } else {
                icon_pause().size(17).color(TEXT)
            },
            if state.paused { "Resume" } else { "Pause" },
            Message::TogglePause,
        ),
        icon_button(
            icon_trash_2().size(17).color(TEXT),
            "Clear history",
            Message::ClearHistory,
        ),
        icon_button(
            if state.collapsed {
                icon_chevron_down().size(18).color(TEXT)
            } else {
                icon_chevron_up().size(18).color(TEXT)
            },
            if state.collapsed {
                "Expand"
            } else {
                "Collapse"
            },
            Message::ToggleCollapsed,
        ),
        icon_button(icon_x().size(18).color(TEXT), "Close", Message::Close),
    ]
    .spacing(6)
    .align_y(Alignment::Center);

    let header = row![
        text("mague-rc").size(18).color(TEXT),
        Space::new().width(Length::Fill),
        container(Space::new())
            .width(8)
            .height(8)
            .style(move |_| dot_style(status_color)),
        text(&state.status).size(13).color(MUTED),
        controls,
    ]
    .spacing(9)
    .align_y(Alignment::Center)
    .width(Length::Fill)
    .height(36);

    if state.collapsed {
        return container(header)
            .padding([8, 12])
            .width(Length::Fill)
            .height(Length::Fill)
            .style(panel_style)
            .into();
    }

    let question = if state.snapshot.current_transcript.is_empty() {
        String::from("Listening for a question")
    } else {
        state.snapshot.current_transcript.clone()
    };
    let answer = if state.snapshot.current_answer.is_empty() {
        if state.snapshot.llm_status == WorkerStatus::Working {
            String::from("Thinking")
        } else {
            String::from("Answer will appear here")
        }
    } else {
        state.snapshot.current_answer.clone()
    };

    let question_section = column![
        text("QUESTION").size(11).color(CYAN),
        text(question).size(15).color(TEXT).width(Length::Fill),
    ]
    .spacing(7)
    .width(Length::Fill);

    let answer_section = column![
        text("ANSWER").size(11).color(MUTED),
        scrollable(text(answer).size(16).color(TEXT).width(Length::Fill))
            .height(Length::Fill)
            .width(Length::Fill),
    ]
    .spacing(8)
    .height(Length::Fill)
    .width(Length::Fill);

    let mut footer = row![text(status_detail(state)).size(12).color(MUTED)]
        .width(Length::Fill)
        .align_y(Alignment::Center);
    if state.snapshot.audio_queue_len > 0 || state.snapshot.llm_queue_len > 0 {
        footer = footer.push(Space::new().width(Length::Fill)).push(
            text(format!(
                "audio {}  |  answers {}",
                state.snapshot.audio_queue_len, state.snapshot.llm_queue_len
            ))
            .size(12)
            .color(MUTED),
        );
    }

    let content = column![
        header,
        rule::horizontal(1).style(|_| rule::Style {
            color: BORDER,
            radius: border::Radius::default(),
            fill_mode: rule::FillMode::Full,
            snap: true,
        }),
        question_section,
        rule::horizontal(1).style(|_| rule::Style {
            color: BORDER,
            radius: border::Radius::default(),
            fill_mode: rule::FillMode::Full,
            snap: true,
        }),
        answer_section,
        footer,
    ]
    .spacing(13)
    .padding(16)
    .width(Length::Fill)
    .height(Length::Fill);

    container(content)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(panel_style)
        .into()
}

fn icon_button<'a>(
    icon: iced::widget::Text<'a>,
    label: &'a str,
    message: Message,
) -> Element<'a, Message> {
    tooltip(
        button(icon)
            .width(34)
            .height(34)
            .padding(0)
            .on_press(message)
            .style(icon_button_style),
        container(text(label).size(12).color(TEXT))
            .padding([6, 8])
            .style(tooltip_style),
        tooltip::Position::Bottom,
    )
    .gap(6)
    .into()
}

fn app_style(_state: &OverlayState, _theme: &Theme) -> iced::theme::Style {
    iced::theme::Style {
        background_color: TRANSPARENT,
        text_color: TEXT,
    }
}

fn app_theme(_state: &OverlayState) -> Theme {
    Theme::Dark
}

fn panel_style(_theme: &Theme) -> container::Style {
    container::Style {
        text_color: Some(TEXT),
        background: Some(Background::Color(PANEL)),
        border: Border {
            color: BORDER,
            width: 1.0,
            radius: 8.0.into(),
        },
        shadow: iced::Shadow {
            color: Color::from_rgba(0.0, 0.0, 0.0, 0.34),
            offset: iced::Vector::new(0.0, 6.0),
            blur_radius: 18.0,
        },
        ..container::Style::default()
    }
}

fn tooltip_style(_theme: &Theme) -> container::Style {
    container::Style {
        text_color: Some(TEXT),
        background: Some(Background::Color(SURFACE)),
        border: Border {
            color: BORDER,
            width: 1.0,
            radius: 6.0.into(),
        },
        ..container::Style::default()
    }
}

fn dot_style(color: Color) -> container::Style {
    container::Style {
        background: Some(Background::Color(color)),
        border: Border {
            radius: 4.0.into(),
            ..Border::default()
        },
        ..container::Style::default()
    }
}

fn icon_button_style(theme: &Theme, status: button::Status) -> button::Style {
    let mut style = button::text(theme, status);
    style.text_color = TEXT;
    style.background = match status {
        button::Status::Hovered | button::Status::Pressed => Some(Background::Color(SURFACE)),
        button::Status::Active | button::Status::Disabled => None,
    };
    style.border = Border {
        color: if matches!(status, button::Status::Hovered) {
            BORDER
        } else {
            TRANSPARENT
        },
        width: 1.0,
        radius: 6.0.into(),
    };
    style
}

fn status_color(state: &OverlayState) -> Color {
    if state.snapshot.last_error.is_some() {
        return RED;
    }
    if state.paused
        || matches!(
            state.snapshot.stt_status,
            ConnectionStatus::Connecting | ConnectionStatus::Reconnecting
        )
        || state.snapshot.llm_status == WorkerStatus::Working
    {
        return AMBER;
    }
    if state.snapshot.listening {
        GREEN
    } else {
        MUTED
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

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> OverlayState {
        let (_event_sender, event_receiver) = mpsc::channel();
        let (command_sender, _command_receiver) = tokio_mpsc::unbounded_channel();
        OverlayState {
            snapshot: AppSnapshot::default(),
            status: String::from("Starting"),
            paused: false,
            collapsed: false,
            stopping: false,
            events: event_receiver,
            commands: command_sender,
        }
    }

    #[test]
    fn applies_streaming_output_to_overlay_state() {
        let mut state = state();
        state.apply_output(OutputEvent::Status(StatusMessage {
            kind: StatusKind::Listening,
            text: String::from("connected"),
        }));
        state.apply_output(OutputEvent::Transcript(crate::events::TranscriptView {
            sequence: 1,
            text: String::from("What is ownership?"),
        }));

        assert_eq!(state.status, "Listening");
        assert!(state.snapshot.listening);
        assert_eq!(state.snapshot.current_transcript, "What is ownership?");
    }

    #[test]
    fn stopped_status_requests_ui_exit() {
        let mut state = state();
        assert!(state.apply_output(OutputEvent::Status(StatusMessage {
            kind: StatusKind::Stopped,
            text: String::from("stopped"),
        })));
    }
}
