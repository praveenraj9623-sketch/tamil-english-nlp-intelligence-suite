from src.preprocessing import clean_text, lowercase_english_tokens


def test_clean_text_preserves_tamil_unicode_and_lowercases_english() -> None:
    tamil = "\u0b87\u0ba8\u0bcd\u0ba4"
    cleaned = clean_text(f"{tamil} SERVICE romba NALLA")

    assert tamil in cleaned
    assert "service" in cleaned
    assert "nalla" in cleaned
    assert "SERVICE" not in cleaned


def test_clean_text_removes_urls_mentions_and_extra_whitespace() -> None:
    cleaned = clean_text("Hello   @support visit https://example.com/path   now")

    assert "@support" not in cleaned
    assert "https://example.com" not in cleaned
    assert cleaned == "hello visit now"


def test_lowercase_english_tokens_does_not_change_tamil() -> None:
    tamil = "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd"
    assert lowercase_english_tokens(f"{tamil} HELLO") == f"{tamil} hello"
