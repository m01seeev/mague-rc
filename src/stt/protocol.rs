use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::events::DeepgramEvent;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("invalid Deepgram JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),

    #[error("Deepgram message does not contain a string type")]
    MissingType,
}

#[derive(Deserialize)]
struct ResultsMessage {
    #[serde(default)]
    is_final: bool,
    #[serde(default)]
    speech_final: bool,
    start: Option<f64>,
    duration: Option<f64>,
    channel: Channel,
}

#[derive(Deserialize)]
struct Channel {
    #[serde(default)]
    alternatives: Vec<Alternative>,
}

#[derive(Deserialize)]
struct Alternative {
    #[serde(default)]
    transcript: String,
}

pub fn parse_server_message(message: &str) -> Result<Option<DeepgramEvent>, ProtocolError> {
    let value: Value = serde_json::from_str(message)?;
    let message_type = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or(ProtocolError::MissingType)?;

    match message_type {
        "Results" => {
            let result: ResultsMessage = serde_json::from_value(value)?;
            let text = result
                .channel
                .alternatives
                .into_iter()
                .next()
                .map(|alternative| alternative.transcript.trim().to_owned())
                .unwrap_or_default();
            Ok(Some(DeepgramEvent::Transcript {
                text,
                is_final: result.is_final,
                speech_final: result.speech_final,
                audio_start_ms: result.start.and_then(seconds_to_ms),
                audio_duration_ms: result.duration.and_then(seconds_to_ms),
            }))
        }
        "SpeechStarted" => Ok(Some(DeepgramEvent::SpeechStarted {
            audio_timestamp_ms: value
                .get("timestamp")
                .and_then(Value::as_f64)
                .and_then(seconds_to_ms),
        })),
        "UtteranceEnd" => Ok(Some(DeepgramEvent::UtteranceEnd {
            last_word_end_ms: value
                .get("last_word_end")
                .and_then(Value::as_f64)
                .and_then(seconds_to_ms),
        })),
        "Metadata" => Ok(Some(DeepgramEvent::Metadata)),
        "Error" => Ok(Some(DeepgramEvent::Error(error_description(&value)))),
        _ => Ok(None),
    }
}

fn seconds_to_ms(seconds: f64) -> Option<u64> {
    if !seconds.is_finite() || seconds.is_sign_negative() {
        return None;
    }
    Some((seconds * 1_000.0).round() as u64)
}

fn error_description(value: &Value) -> String {
    let code = value.get("code").and_then(Value::as_str);
    let description = value
        .get("description")
        .or_else(|| value.get("message"))
        .and_then(Value::as_str);

    match (code, description) {
        (Some(code), Some(description)) => format!("{code}: {description}"),
        (Some(code), None) => code.to_owned(),
        (None, Some(description)) => description.to_owned(),
        (None, None) => "unspecified Deepgram error".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_final_transcript() {
        let message = r#"{
            "type": "Results",
            "is_final": true,
            "speech_final": false,
            "channel": {
                "alternatives": [{"transcript": "  Привет, мир.  "}]
            }
        }"#;

        let event = parse_server_message(message)
            .expect("message must parse")
            .expect("event must be produced");

        assert_eq!(
            event,
            DeepgramEvent::Transcript {
                text: "Привет, мир.".to_owned(),
                is_final: true,
                speech_final: false,
                audio_start_ms: None,
                audio_duration_ms: None,
            }
        );
    }

    #[test]
    fn parses_interim_transcript() {
        let message = r#"{
            "type": "Results",
            "is_final": false,
            "speech_final": false,
            "channel": {"alternatives": [{"transcript": "hash map"}]}
        }"#;

        let event = parse_server_message(message)
            .expect("message must parse")
            .expect("event must be produced");

        assert!(matches!(
            event,
            DeepgramEvent::Transcript {
                is_final: false,
                ..
            }
        ));
    }

    #[test]
    fn parses_transcript_audio_position() {
        let event = parse_server_message(
            r#"{
                "type": "Results",
                "start": 1.25,
                "duration": 0.75,
                "channel": {"alternatives": [{"transcript": "test"}]}
            }"#,
        )
        .expect("message must parse")
        .expect("event must be produced");

        assert!(matches!(
            event,
            DeepgramEvent::Transcript {
                audio_start_ms: Some(1_250),
                audio_duration_ms: Some(750),
                ..
            }
        ));
    }

    #[test]
    fn parses_control_events() {
        assert_eq!(
            parse_server_message(r#"{"type":"SpeechStarted","timestamp":0.5}"#)
                .expect("message must parse"),
            Some(DeepgramEvent::SpeechStarted {
                audio_timestamp_ms: Some(500),
            })
        );
        assert_eq!(
            parse_server_message(r#"{"type":"UtteranceEnd","last_word_end":2.5}"#)
                .expect("message must parse"),
            Some(DeepgramEvent::UtteranceEnd {
                last_word_end_ms: Some(2_500),
            })
        );
        assert_eq!(
            parse_server_message(r#"{"type":"Metadata"}"#).expect("message must parse"),
            Some(DeepgramEvent::Metadata)
        );
    }

    #[test]
    fn ignores_unknown_event_types() {
        let event = parse_server_message(r#"{"type":"FutureEvent"}"#).expect("message must parse");

        assert_eq!(event, None);
    }

    #[test]
    fn reports_server_error_without_losing_code() {
        let event =
            parse_server_message(r#"{"type":"Error","code":"NET-0001","description":"timeout"}"#)
                .expect("message must parse");

        assert_eq!(
            event,
            Some(DeepgramEvent::Error("NET-0001: timeout".to_owned()))
        );
    }
}
