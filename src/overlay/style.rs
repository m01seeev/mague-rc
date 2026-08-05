use gtk::{CssProvider, STYLE_PROVIDER_PRIORITY_APPLICATION, gdk};

pub(super) fn install_css() {
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

        .mode-tabs {
            background: transparent;
        }

        .mode-tab {
            min-height: 28px;
            padding: 2px 12px;
            color: #777f89;
            background: transparent;
            border: 1px solid #2b3036;
            border-radius: 4px;
            box-shadow: none;
            font-family: "JetBrains Mono", monospace;
            font-size: 10px;
            font-weight: 600;
        }

        .mode-tab:hover {
            color: #edf0f2;
            border-color: #4b525c;
        }

        .mode-tab.active {
            color: #52bacf;
            background: #162327;
            border-color: #316773;
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

        .coding-page {
            background: rgba(9, 11, 13, 0.42);
        }

        .coding-workspace {
            border-top: 1px solid #252a30;
        }

        .coding-context {
            padding-top: 14px;
        }

        .coding-editor {
            padding-top: 14px;
        }

        .coding-state-block {
            padding: 9px 11px;
            background: rgba(25, 28, 32, 0.88);
            border-left: 2px solid #316773;
            border-radius: 3px;
        }

        .coding-talk-block {
            border-left-color: #c9974d;
            background: rgba(35, 31, 25, 0.90);
        }

        .coding-revision,
        .coding-language,
        .coding-change-note {
            font-family: "JetBrains Mono", monospace;
            font-size: 10px;
        }

        .coding-revision {
            color: #52bacf;
            font-weight: 600;
        }

        .coding-language,
        .coding-change-note {
            color: #777f89;
        }

        .coding-explanation,
        .coding-summary {
            color: #d9dde1;
            font-family: "Noto Sans", sans-serif;
            font-size: 13px;
        }

        .coding-explanation {
            color: #f0dfc2;
            font-size: 14px;
        }

        .coding-text-scroll {
            background: transparent;
        }

        .coding-code-scroll {
            margin: 0 16px 14px 14px;
            border: 1px solid #2b3036;
            border-radius: 4px;
        }

        .coding-code,
        .coding-code text {
            color: #d9e1e5;
            background: #0b0d0f;
            font-family: "JetBrains Mono", monospace;
            font-size: 12px;
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
