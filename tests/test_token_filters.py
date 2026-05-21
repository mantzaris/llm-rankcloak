from rankcloak.token_filters import is_safe_text_token_piece


def test_safe_text_filter_rejects_artifact_strings():
    for piece in [
        "```",
        "\\section",
        "\\frac",
        "&amp",
        "http",
        "www.",
        "[Name]",
        "[original author]",
        "<div>",
    ]:
        assert not is_safe_text_token_piece(piece)


def test_safe_text_filter_allows_ordinary_prose_fragments():
    for piece in [
        " lentils",
        " onion",
        " simmer",
        " the",
        " and",
        " garden",
        " soil",
    ]:
        assert is_safe_text_token_piece(piece)
