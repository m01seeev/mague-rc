const DANGLING_FINAL_WORDS: &[&str] = &[
    "а",
    "без",
    "в",
    "во",
    "для",
    "за",
    "если",
    "и",
    "из",
    "или",
    "к",
    "ко",
    "которого",
    "котором",
    "которой",
    "которому",
    "которым",
    "которыми",
    "которую",
    "которые",
    "которых",
    "который",
    "которая",
    "которое",
    "какая",
    "какие",
    "каким",
    "какими",
    "какого",
    "какое",
    "какой",
    "какую",
    "каких",
    "либо",
    "между",
    "на",
    "над",
    "не",
    "ни",
    "но",
    "о",
    "об",
    "обо",
    "от",
    "перед",
    "по",
    "под",
    "пока",
    "потому",
    "при",
    "про",
    "с",
    "со",
    "у",
    "хотя",
    "через",
    "что",
    "чего",
    "чем",
    "чтобы",
];

const DANGLING_FINAL_PHRASES: &[&[&str]] = &[
    &["в", "каких"],
    &["в", "каком"],
    &["для", "того", "чтобы"],
    &["за", "счет"],
    &["за", "счёт"],
    &["и", "какая"],
    &["и", "какие"],
    &["и", "какой"],
    &["и", "какую"],
    &["если", "у", "нас"],
    &["потому", "что"],
];

const REQUEST_WORDS: &[&str] = &[
    "объясни",
    "объясните",
    "опиши",
    "опишите",
    "покажи",
    "покажите",
    "расскажи",
    "расскажите",
    "сравни",
    "сравните",
];

const QUESTION_WORDS: &[&str] = &[
    "где",
    "зачем",
    "как",
    "какая",
    "какие",
    "каким",
    "какими",
    "какого",
    "какое",
    "какой",
    "какую",
    "каких",
    "когда",
    "почему",
    "сколько",
    "чего",
    "чем",
    "что",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BoundaryDeferral {
    DanglingSuffix,
    Introduction,
    Setup,
    ShortFragment,
}

impl BoundaryDeferral {
    pub(super) fn reason(self) -> &'static str {
        match self {
            Self::DanglingSuffix => "dangling_suffix",
            Self::Introduction => "introduction",
            Self::Setup => "setup",
            Self::ShortFragment => "short_fragment",
        }
    }
}

pub(super) fn boundary_deferral(text: &str) -> Option<BoundaryDeferral> {
    let words = normalized_words(text);
    if words.is_empty() || ends_with_question_mark(text) {
        return None;
    }
    if DANGLING_FINAL_PHRASES
        .iter()
        .any(|suffix| ends_with_words(&words, suffix))
        || words.last().is_some_and(|word| {
            DANGLING_FINAL_WORDS
                .iter()
                .any(|candidate| word == candidate)
        })
    {
        return Some(BoundaryDeferral::DanglingSuffix);
    }
    if is_introduction(&words) && !has_request_word(&words) {
        return Some(BoundaryDeferral::Introduction);
    }
    if is_setup(&words) && !has_request_word(&words) {
        return Some(BoundaryDeferral::Setup);
    }
    if words.len() <= 6 && !has_question_signal(&words) {
        return Some(BoundaryDeferral::ShortFragment);
    }
    None
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric() && character != '-')
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn ends_with_words(words: &[String], suffix: &[&str]) -> bool {
    words.len() >= suffix.len()
        && words[words.len() - suffix.len()..]
            .iter()
            .zip(suffix)
            .all(|(word, candidate)| word == candidate)
}

fn ends_with_question_mark(text: &str) -> bool {
    text.trim_end_matches(|character: char| {
        character.is_whitespace() || matches!(character, '"' | '\'' | ')' | ']')
    })
    .ends_with('?')
}

fn is_setup(words: &[String]) -> bool {
    words
        .iter()
        .any(|word| matches!(word.as_str(), "допустим" | "предположим" | "представим"))
        || contains_words(words, &["у", "нас", "есть"])
}

fn is_introduction(words: &[String]) -> bool {
    words.iter().any(|word| word == "давайте")
        && words
            .iter()
            .any(|word| matches!(word.as_str(), "начнем" | "начнём" | "начнемте" | "начнёмте"))
}

fn contains_words(words: &[String], needle: &[&str]) -> bool {
    words.len() >= needle.len()
        && words
            .windows(needle.len())
            .any(|window| ends_with_words(window, needle))
}

fn has_request_word(words: &[String]) -> bool {
    words
        .iter()
        .any(|word| REQUEST_WORDS.iter().any(|candidate| word == candidate))
}

fn has_question_signal(words: &[String]) -> bool {
    has_request_word(words)
        || words
            .iter()
            .any(|word| QUESTION_WORDS.iter().any(|candidate| word == candidate))
}
