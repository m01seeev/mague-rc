use crate::events::TranscriptChunk;

#[derive(Debug)]
pub struct TranscriptWindowAssembler {
    min_utterance_chars: usize,
    pending_finals: Vec<String>,
    current_interim: String,
    sent_interim: String,
    next_sequence: u64,
}

impl TranscriptWindowAssembler {
    pub fn new(min_utterance_chars: usize) -> Self {
        Self {
            min_utterance_chars,
            pending_finals: Vec::new(),
            current_interim: String::new(),
            sent_interim: String::new(),
            next_sequence: 0,
        }
    }

    pub fn push_transcript(&mut self, text: &str, is_final: bool) {
        let text = normalize(text);

        if is_final {
            if let Some(novel) = novel_text(&self.sent_interim, &text) {
                self.pending_finals.push(novel);
            }
            self.current_interim.clear();
            self.sent_interim.clear();
        } else {
            self.current_interim = text;
        }
    }

    pub fn flush(&mut self) -> Option<TranscriptChunk> {
        if !self.pending_finals.is_empty() {
            let text = self.pending_finals.join(" ");
            if char_count(&text) < self.min_utterance_chars {
                return None;
            }

            self.pending_finals.clear();
            return Some(self.chunk(text));
        }

        let text = novel_text(&self.sent_interim, &self.current_interim)?;
        if char_count(&text) < self.min_utterance_chars {
            return None;
        }

        self.sent_interim.clone_from(&self.current_interim);
        Some(self.chunk(text))
    }

    pub fn finish(&mut self) -> Option<TranscriptChunk> {
        let mut parts = std::mem::take(&mut self.pending_finals);
        if let Some(interim) = novel_text(&self.sent_interim, &self.current_interim) {
            parts.push(interim);
        }
        self.current_interim.clear();
        self.sent_interim.clear();

        let text = parts.join(" ");
        if char_count(&text) < self.min_utterance_chars {
            return None;
        }

        Some(self.chunk(text))
    }

    fn chunk(&mut self, text: String) -> TranscriptChunk {
        let chunk = TranscriptChunk {
            sequence: self.next_sequence,
            text,
        };
        self.next_sequence = self.next_sequence.saturating_add(1);
        chunk
    }
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn novel_text(previous: &str, current: &str) -> Option<String> {
    if current.is_empty() || current == previous {
        return None;
    }
    if previous.is_empty() {
        return Some(current.to_owned());
    }

    current
        .strip_prefix(previous)
        .filter(|tail| {
            tail.chars()
                .next()
                .is_some_and(|character| character.is_whitespace())
        })
        .map(str::trim)
        .filter(|tail| !tail.is_empty())
        .map(str::to_owned)
        .or_else(|| Some(current.to_owned()))
}

fn char_count(text: &str) -> usize {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assembler() -> TranscriptWindowAssembler {
        TranscriptWindowAssembler::new(3)
    }

    fn text(chunk: Option<TranscriptChunk>) -> Option<String> {
        chunk.map(|chunk| chunk.text)
    }

    #[test]
    fn flushes_only_final_text() {
        let mut assembler = assembler();
        assembler.push_transcript("Что такое HashMap?", true);

        assert_eq!(
            text(assembler.flush()),
            Some("Что такое HashMap?".to_owned())
        );
    }

    #[test]
    fn flushes_only_new_tail_of_growing_interim() {
        let mut assembler = assembler();
        assembler.push_transcript("расскажите про hash", false);
        assert_eq!(
            text(assembler.flush()),
            Some("расскажите про hash".to_owned())
        );

        assembler.push_transcript("расскажите про hash map и устройство", false);
        assert_eq!(text(assembler.flush()), Some("map и устройство".to_owned()));
    }

    #[test]
    fn matching_final_does_not_repeat_sent_interim() {
        let mut assembler = assembler();
        assembler.push_transcript("расскажите про HashMap", false);
        assert!(assembler.flush().is_some());

        assembler.push_transcript("расскажите про HashMap", true);

        assert_eq!(assembler.flush(), None);
    }

    #[test]
    fn corrected_final_is_sent_in_full() {
        let mut assembler = assembler();
        assembler.push_transcript("расскажите про HashSet", false);
        assert!(assembler.flush().is_some());

        assembler.push_transcript("расскажите про HashMap", true);

        assert_eq!(
            text(assembler.flush()),
            Some("расскажите про HashMap".to_owned())
        );
    }

    #[test]
    fn joins_multiple_finals_inside_one_window() {
        let mut assembler = assembler();
        assembler.push_transcript("Как устроен HashMap?", true);
        assembler.push_transcript("Почему нужен hashCode?", true);

        assert_eq!(
            text(assembler.flush()),
            Some("Как устроен HashMap? Почему нужен hashCode?".to_owned())
        );
    }

    #[test]
    fn empty_window_produces_no_chunk() {
        assert_eq!(assembler().flush(), None);
    }

    #[test]
    fn shutdown_flush_returns_accumulated_text() {
        let mut assembler = assembler();
        assembler.push_transcript("незаконченный вопрос", false);

        assert_eq!(
            text(assembler.finish()),
            Some("незаконченный вопрос".to_owned())
        );
    }

    #[test]
    fn shutdown_flush_combines_pending_final_and_new_interim() {
        let mut assembler = assembler();
        assembler.push_transcript("подтвержденная часть", true);
        assembler.push_transcript("новый незаконченный хвост", false);

        assert_eq!(
            text(assembler.finish()),
            Some("подтвержденная часть новый незаконченный хвост".to_owned())
        );
        assert_eq!(assembler.finish(), None);
    }

    #[test]
    fn repeated_growing_prefix_is_not_duplicated() {
        let mut assembler = assembler();
        assembler.push_transcript("как работает", false);
        assert_eq!(text(assembler.flush()), Some("как работает".to_owned()));

        assembler.push_transcript("как работает сборщик мусора", false);
        assert_eq!(text(assembler.flush()), Some("сборщик мусора".to_owned()));
        assert_eq!(assembler.flush(), None);
    }

    #[test]
    fn short_tail_waits_for_more_interim_text() {
        let mut assembler = TranscriptWindowAssembler::new(5);
        assembler.push_transcript("что такое hash", false);
        assert!(assembler.flush().is_some());

        assembler.push_transcript("что такое hash map", false);
        assert_eq!(assembler.flush(), None);

        assembler.push_transcript("что такое hash map внутри", false);
        assert_eq!(text(assembler.flush()), Some("map внутри".to_owned()));
    }

    #[test]
    fn final_has_priority_over_new_interim() {
        let mut assembler = assembler();
        assembler.push_transcript("первая часть", true);
        assembler.push_transcript("следующая часть вопроса", false);

        assert_eq!(text(assembler.flush()), Some("первая часть".to_owned()));
        assert_eq!(
            text(assembler.flush()),
            Some("следующая часть вопроса".to_owned())
        );
    }
}
