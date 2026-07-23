use std::{
    io::{self, IsTerminal},
    process::{Command, Stdio},
};

use tracing::warn;

pub struct TerminalEchoGuard {
    original_state: Option<String>,
}

impl TerminalEchoGuard {
    pub fn hide_control_characters() -> Self {
        if !io::stdin().is_terminal() {
            return Self {
                original_state: None,
            };
        }

        match disable_control_character_echo() {
            Ok(original_state) => Self {
                original_state: Some(original_state),
            },
            Err(error) => {
                warn!(
                    module = "control",
                    event = "terminal_echo_unchanged",
                    error = %error,
                    "could not hide terminal control-character echo"
                );
                Self {
                    original_state: None,
                }
            }
        }
    }
}

impl Drop for TerminalEchoGuard {
    fn drop(&mut self) {
        let Some(original_state) = self.original_state.take() else {
            return;
        };

        if let Err(error) = run_stty(&original_state) {
            warn!(
                module = "control",
                event = "terminal_restore_failed",
                error = %error,
                "could not restore terminal settings"
            );
        }
    }
}

fn disable_control_character_echo() -> Result<String, io::Error> {
    let output = Command::new("stty")
        .arg("-g")
        .stdin(Stdio::inherit())
        .stderr(Stdio::null())
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "stty -g exited with status {}",
            output.status
        )));
    }

    let original_state = String::from_utf8(output.stdout)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let original_state = original_state.trim().to_owned();
    if original_state.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "stty returned an empty terminal state",
        ));
    }

    run_stty("-echoctl")?;
    Ok(original_state)
}

fn run_stty(argument: &str) -> Result<(), io::Error> {
    let status = Command::new("stty")
        .arg(argument)
        .stdin(Stdio::inherit())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;

    if !status.success() {
        return Err(io::Error::other(format!(
            "stty {argument} exited with status {status}"
        )));
    }
    Ok(())
}
