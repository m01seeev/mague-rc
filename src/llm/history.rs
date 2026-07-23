use std::collections::VecDeque;

use crate::{events::Mode, llm::ChatMessage};

#[derive(Clone, Debug, PartialEq, Eq)]
struct HistoryPair {
    user: String,
    assistant: String,
}

#[derive(Debug)]
struct ChatHistory {
    maximum_pairs: usize,
    pairs: VecDeque<HistoryPair>,
}

impl ChatHistory {
    fn new(maximum_pairs: usize) -> Self {
        Self {
            maximum_pairs,
            pairs: VecDeque::with_capacity(maximum_pairs),
        }
    }

    fn record(&mut self, user: String, assistant: String) {
        if self.pairs.len() == self.maximum_pairs {
            self.pairs.pop_front();
        }
        self.pairs.push_back(HistoryPair { user, assistant });
    }

    fn messages(&self) -> Vec<ChatMessage> {
        self.pairs
            .iter()
            .flat_map(|pair| {
                [
                    ChatMessage::user(pair.user.clone()),
                    ChatMessage::assistant(pair.assistant.clone()),
                ]
            })
            .collect()
    }

    fn clear(&mut self) {
        self.pairs.clear();
    }
}

#[derive(Debug)]
pub struct ConversationHistories {
    voice: ChatHistory,
    ocr: ChatHistory,
    separate: bool,
}

impl ConversationHistories {
    pub fn new(maximum_pairs: usize, separate: bool) -> Self {
        Self {
            voice: ChatHistory::new(maximum_pairs),
            ocr: ChatHistory::new(maximum_pairs),
            separate,
        }
    }

    pub fn messages(&self, mode: Mode) -> Vec<ChatMessage> {
        self.history(mode).messages()
    }

    pub fn record(&mut self, mode: Mode, user: String, assistant: String) {
        self.history_mut(mode).record(user, assistant);
    }

    pub fn clear(&mut self) {
        self.voice.clear();
        self.ocr.clear();
    }

    fn history(&self, mode: Mode) -> &ChatHistory {
        match (self.separate, mode) {
            (true, Mode::Ocr) => &self.ocr,
            _ => &self.voice,
        }
    }

    fn history_mut(&mut self, mode: Mode) -> &mut ChatHistory {
        match (self.separate, mode) {
            (true, Mode::Ocr) => &mut self.ocr,
            _ => &mut self.voice,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::llm::ChatRole;

    use super::*;

    fn contents(messages: Vec<ChatMessage>) -> Vec<(ChatRole, String)> {
        messages
            .into_iter()
            .map(|message| (message.role, message.content))
            .collect()
    }

    #[test]
    fn retains_only_the_latest_pairs() {
        let mut histories = ConversationHistories::new(2, true);
        histories.record(Mode::Voice, "u1".to_owned(), "a1".to_owned());
        histories.record(Mode::Voice, "u2".to_owned(), "a2".to_owned());
        histories.record(Mode::Voice, "u3".to_owned(), "a3".to_owned());

        assert_eq!(
            contents(histories.messages(Mode::Voice)),
            vec![
                (ChatRole::User, "u2".to_owned()),
                (ChatRole::Assistant, "a2".to_owned()),
                (ChatRole::User, "u3".to_owned()),
                (ChatRole::Assistant, "a3".to_owned()),
            ]
        );
    }

    #[test]
    fn keeps_voice_and_ocr_histories_separate() {
        let mut histories = ConversationHistories::new(4, true);
        histories.record(Mode::Voice, "voice".to_owned(), "voice answer".to_owned());
        histories.record(Mode::Ocr, "ocr".to_owned(), "ocr answer".to_owned());

        assert_eq!(histories.messages(Mode::Voice).len(), 2);
        assert_eq!(histories.messages(Mode::Ocr).len(), 2);
        assert_eq!(
            histories.messages(Mode::Voice)[0].content,
            "voice".to_owned()
        );
        assert_eq!(histories.messages(Mode::Ocr)[0].content, "ocr".to_owned());
    }

    #[test]
    fn shares_history_when_separation_is_disabled() {
        let mut histories = ConversationHistories::new(4, false);
        histories.record(Mode::Voice, "voice".to_owned(), "answer".to_owned());

        assert_eq!(
            histories.messages(Mode::Ocr),
            histories.messages(Mode::Voice)
        );
    }

    #[test]
    fn clears_all_histories() {
        let mut histories = ConversationHistories::new(4, true);
        histories.record(Mode::Voice, "voice".to_owned(), "answer".to_owned());
        histories.record(Mode::Ocr, "ocr".to_owned(), "answer".to_owned());

        histories.clear();

        assert!(histories.messages(Mode::Voice).is_empty());
        assert!(histories.messages(Mode::Ocr).is_empty());
    }
}
