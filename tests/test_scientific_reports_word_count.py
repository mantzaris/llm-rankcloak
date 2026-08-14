import shutil

import pytest

from tools.count_scientific_reports import (
    _table_words,
    _texcount,
    command_arguments,
    environment_blocks,
    split_sections,
    strip_comments,
)


pytestmark = pytest.mark.skipif(
    shutil.which("texcount") is None, reason="TeXcount is not installed"
)


def test_balanced_commands_and_sections_preserve_nested_text():
    source = r"""
    \section*{Introduction}
    Alpha \textbf{nested {words}} omega.
    \section*{Results}
    Result text.
    """

    arguments = list(command_arguments(source, "textbf"))
    assert arguments[0][2] == "nested {words}"
    sections = split_sections(source)
    assert "Alpha" in sections["Introduction"]
    assert "Result text" in sections["Results"]


def test_comment_stripping_preserves_escaped_percent():
    source = "kept \\% value % removed\nnext\n"
    assert strip_comments(source) == "kept \\% value \nnext\n"


def test_texcount_separates_prose_caption_and_table_body():
    source = r"""
    Narrative has three words.
    \begin{table}
      \begin{tabular}{ll}
      Header one & Header two \\
      Value one & Value two \\
      \end{tabular}
      \caption{A four word table caption.}
    \end{table}
    """
    count = _texcount(source, preamble=r"\documentclass{article}")
    assert count.text_words == 4
    assert count.caption_words == 5

    table = environment_blocks(source, "table")[0]
    body_words, caption_words = _table_words(table)
    assert body_words == 8
    assert caption_words == 5
