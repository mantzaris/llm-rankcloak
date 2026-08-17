#!/usr/bin/env python3
"""Report Scientific Reports manuscript word and display-item counts.

The counter uses TeXcount's LaTeX-aware definition of a prose word.  It then
separates material that TeXcount normally groups together so that the journal
main-text count, table-inclusive conservative count, and individual figure
legends can be audited independently.

By design, section headings, displayed mathematics, algorithm pseudocode,
bibliography entries, and front/back-matter declarations are not prose words.
The conservative count adds visible table bodies and table captions.  Figure
legends remain separate, as required by the Scientific Reports word limit.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN = REPO_ROOT / "paperV2" / "scientific_reports" / "main2.tex"
DEFAULT_SUPPLEMENT = (
    REPO_ROOT / "paperV2" / "scientific_reports" / "supplementary2.tex"
)

COUNTED_MAIN_SECTIONS = (
    "Introduction",
    "Results",
    "Discussion",
    "Responsible use and limitations",
)


@dataclass(frozen=True)
class TexCount:
    text_words: int
    header_words: int
    caption_words: int
    headers: int
    floats: int
    inline_math: int
    displayed_math: int


@dataclass(frozen=True)
class EnvironmentBlock:
    environment: str
    start: int
    end: int
    body: str
    source: str


@dataclass(frozen=True)
class FigureLegend:
    number: int
    label: str
    image: str
    words: int


@dataclass(frozen=True)
class ManuscriptReport:
    main_file: str
    supplementary_file: str
    title: str
    title_words: int
    abstract_words: int
    introduction_words: int
    results_words: int
    discussion_and_limitations_words: int
    journal_main_text_words: int
    conservative_main_text_words: int
    methods_words: int
    figure_legends: List[FigureLegend]
    supplementary_information_words: int
    main_figures: int
    main_tables: int
    main_display_items: int
    diagnostics: Dict[str, int]


def strip_comments(source: str) -> str:
    """Remove unescaped LaTeX comments while preserving line boundaries."""

    output: List[str] = []
    for line in source.splitlines(keepends=True):
        cut = len(line)
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                cut = index
                break
        kept = line[:cut]
        if line.endswith("\n") and not kept.endswith("\n"):
            kept += "\n"
        output.append(kept)
    return "".join(output)


def _balanced_group(source: str, opening: int, left: str, right: str) -> Tuple[str, int]:
    if opening >= len(source) or source[opening] != left:
        raise ValueError("balanced-group parser was not positioned at an opening delimiter")
    depth = 0
    index = opening
    while index < len(source):
        character = source[index]
        escaped = index > 0 and source[index - 1] == "\\"
        if not escaped and character == left:
            depth += 1
        elif not escaped and character == right:
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index], index + 1
        index += 1
    raise ValueError("unbalanced LaTeX group")


def command_arguments(source: str, command: str) -> Iterator[Tuple[int, int, str]]:
    """Yield balanced mandatory arguments for a LaTeX command."""

    pattern = re.compile(r"\\" + re.escape(command) + r"\*?\s*(?:\[[^\]]*\]\s*)?")
    for match in pattern.finditer(source):
        opening = match.end()
        if opening >= len(source) or source[opening] != "{":
            continue
        argument, end = _balanced_group(source, opening, "{", "}")
        yield match.start(), end, argument


def first_command_argument(source: str, command: str) -> str:
    try:
        return next(command_arguments(source, command))[2]
    except StopIteration as error:
        raise ValueError("missing \\{}{{...}} command".format(command)) from error


def environment_blocks(source: str, environment: str) -> List[EnvironmentBlock]:
    """Return non-overlapping blocks for a non-recursive LaTeX environment."""

    begin_pattern = re.compile(
        r"\\begin\s*\{" + re.escape(environment) + r"\}(?:\s*\[[^\]]*\])?"
    )
    end_pattern = re.compile(r"\\end\s*\{" + re.escape(environment) + r"\}")
    blocks: List[EnvironmentBlock] = []
    cursor = 0
    while True:
        begin = begin_pattern.search(source, cursor)
        if begin is None:
            break
        end = end_pattern.search(source, begin.end())
        if end is None:
            raise ValueError("unclosed {} environment".format(environment))
        blocks.append(
            EnvironmentBlock(
                environment=environment,
                start=begin.start(),
                end=end.end(),
                body=source[begin.end() : end.start()],
                source=source[begin.start() : end.end()],
            )
        )
        cursor = end.end()
    return blocks


def _replace_spans(source: str, spans: Iterable[Tuple[int, int]]) -> str:
    result = source
    for start, end in sorted(spans, reverse=True):
        result = result[:start] + "\n" + result[end:]
    return result


def remove_commands(source: str, command: str) -> str:
    return _replace_spans(
        source, ((start, end) for start, end, _ in command_arguments(source, command))
    )


def split_sections(source: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    commands = list(command_arguments(source, "section"))
    for index, (start, end, title_source) in enumerate(commands):
        next_start = commands[index + 1][0] if index + 1 < len(commands) else len(source)
        title = latex_to_label(title_source)
        sections[title] = source[end:next_start]
    return sections


def latex_to_label(source: str) -> str:
    text = source.replace("\\\\", " ")
    text = re.sub(r"\\[A-Za-z@]+\*?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    return " ".join(text.split())


def _texcount(source: str, preamble: str = "") -> TexCount:
    executable = shutil.which("texcount")
    if executable is None:
        raise RuntimeError(
            "texcount is required (it is normally installed with TeX Live)"
        )
    wrapper = (
        preamble.rstrip()
        + "\n\\begin{document}\n"
        + source
        + "\n\\end{document}\n"
    )
    template = "{1}|{2}|{3}|{4}|{5}|{6}|{7}"
    completed = subprocess.run(
        [
            executable,
            "-utf8",
            "-nosub",
            "-quiet",
            "-template={}".format(template),
            "-",
        ],
        input=wrapper,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    matches = re.findall(r"(?m)^(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)$", completed.stdout)
    if completed.returncode != 0 or not matches:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError("texcount failed: {}".format(detail))
    values = [int(value) for value in matches[-1]]
    return TexCount(*values)


def _document_preamble(source: str) -> str:
    marker = re.search(r"\\begin\s*\{document\}", source)
    if marker is None:
        return "\\documentclass{article}"
    preamble = source[: marker.start()]
    for block in environment_blocks(preamble, "abstract"):
        preamble = _replace_spans(preamble, [(block.start, block.end)])
    return preamble


def _caption_text(block: EnvironmentBlock) -> Optional[str]:
    captions = list(command_arguments(block.body, "caption"))
    return captions[0][2] if captions else None


def _caption_words(block: EnvironmentBlock) -> int:
    caption = _caption_text(block)
    return _texcount(caption).text_words if caption is not None else 0


def _table_words(block: EnvironmentBlock) -> Tuple[int, int]:
    """Return (body words, caption words) for a table block."""

    body = remove_commands(block.body, "caption")
    table_preamble = "\n".join(
        [
            "\\documentclass{article}",
            "%TC:envir tabular 1 1",
            "%TC:envir tabular* 1 1",
        ]
    )
    return _texcount(body, preamble=table_preamble).text_words, _caption_words(block)


def _section_prose(section: str, preamble: str) -> TexCount:
    return _texcount(section, preamble=preamble)


def _figure_legends(source: str) -> List[FigureLegend]:
    blocks = environment_blocks(source, "figure") + environment_blocks(source, "figure*")
    blocks.sort(key=lambda block: block.start)
    legends: List[FigureLegend] = []
    for number, block in enumerate(blocks, start=1):
        caption = _caption_text(block) or ""
        labels = list(command_arguments(caption, "label"))
        label = labels[0][2] if labels else "figure-{}".format(number)
        images = list(command_arguments(block.body, "includegraphics"))
        image = images[0][2] if images else ""
        legends.append(
            FigureLegend(
                number=number,
                label=label,
                image=image,
                words=_texcount(caption).text_words,
            )
        )
    return legends


def _tables_in(source: str) -> List[EnvironmentBlock]:
    blocks = environment_blocks(source, "table") + environment_blocks(source, "table*")
    return sorted(blocks, key=lambda block: block.start)


def _all_caption_words(source: str) -> int:
    return sum(_texcount(argument).text_words for _, _, argument in command_arguments(source, "caption"))


def build_report(main_path: Path, supplement_path: Path) -> ManuscriptReport:
    main_source = strip_comments(main_path.read_text(encoding="utf-8"))
    supplement_source = strip_comments(supplement_path.read_text(encoding="utf-8"))

    main_preamble = _document_preamble(main_source)
    supplement_preamble = _document_preamble(supplement_source)
    main_sections = split_sections(main_source)
    supplement_sections = split_sections(supplement_source)

    missing = [name for name in COUNTED_MAIN_SECTIONS if name not in main_sections]
    if "Methods" not in main_sections:
        missing.append("Methods")
    if missing:
        raise ValueError("missing required main section(s): {}".format(", ".join(missing)))

    title_source = first_command_argument(main_source, "title")
    abstract_blocks = environment_blocks(main_source, "abstract")
    if len(abstract_blocks) != 1:
        raise ValueError("expected exactly one main abstract environment")

    section_counts = {
        name: _section_prose(body, main_preamble).text_words
        for name, body in main_sections.items()
    }
    journal_main = sum(section_counts[name] for name in COUNTED_MAIN_SECTIONS)

    counted_table_body = 0
    counted_table_captions = 0
    for section_name in COUNTED_MAIN_SECTIONS:
        for table in _tables_in(main_sections[section_name]):
            body_words, caption_words = _table_words(table)
            counted_table_body += body_words
            counted_table_captions += caption_words

    main_figures = len(environment_blocks(main_source, "figure")) + len(
        environment_blocks(main_source, "figure*")
    )
    main_tables = len(_tables_in(main_source))
    legends = _figure_legends(main_source)

    supplement_abstracts = environment_blocks(supplement_source, "abstract")
    supplement_abstract_words = sum(
        _texcount(block.body).text_words for block in supplement_abstracts
    )
    supplement_prose_words = sum(
        _section_prose(body, supplement_preamble).text_words
        for body in supplement_sections.values()
    )
    supplement_table_body_words = sum(
        _table_words(block)[0] for block in _tables_in(supplement_source)
    )
    supplement_caption_words = _all_caption_words(supplement_source)
    supplement_total = (
        supplement_abstract_words
        + supplement_prose_words
        + supplement_table_body_words
        + supplement_caption_words
    )

    return ManuscriptReport(
        main_file=str(main_path),
        supplementary_file=str(supplement_path),
        title=latex_to_label(title_source),
        title_words=_texcount(title_source).text_words,
        abstract_words=_texcount(abstract_blocks[0].body).text_words,
        introduction_words=section_counts["Introduction"],
        results_words=section_counts["Results"],
        discussion_and_limitations_words=(
            section_counts["Discussion"]
            + section_counts["Responsible use and limitations"]
        ),
        journal_main_text_words=journal_main,
        conservative_main_text_words=(
            journal_main + counted_table_body + counted_table_captions
        ),
        methods_words=section_counts["Methods"],
        figure_legends=legends,
        supplementary_information_words=supplement_total,
        main_figures=main_figures,
        main_tables=main_tables,
        main_display_items=main_figures + main_tables,
        diagnostics={
            "discussion_words": section_counts["Discussion"],
            "responsible_use_and_limitations_words": section_counts[
                "Responsible use and limitations"
            ],
            "data_and_code_availability_words": section_counts.get(
                "Data and code availability", 0
            ),
            "counted_main_table_body_words": counted_table_body,
            "counted_main_table_caption_words": counted_table_captions,
            "results_with_table_text_words": (
                section_counts["Results"]
                + sum(
                    sum(_table_words(table))
                    for table in _tables_in(main_sections["Results"])
                )
            ),
            "main_figure_legend_words": sum(legend.words for legend in legends),
            "supplement_abstract_words": supplement_abstract_words,
            "supplement_prose_words": supplement_prose_words,
            "supplement_table_body_words": supplement_table_body_words,
            "supplement_caption_words": supplement_caption_words,
        },
    )


def _display_report(report: ManuscriptReport) -> None:
    print("Scientific Reports manuscript audit")
    print("Main source: {}".format(report.main_file))
    print("Supplement source: {}".format(report.supplementary_file))
    print()
    print("1. Abstract: {:,} words".format(report.abstract_words))
    print("2. Introduction: {:,} words".format(report.introduction_words))
    print("3. Results: {:,} words".format(report.results_words))
    print(
        "4. Discussion + Responsible use and limitations: {:,} words".format(
            report.discussion_and_limitations_words
        )
    )
    print(
        "5. Journal-defined main text: {:,} words".format(
            report.journal_main_text_words
        )
    )
    print(
        "6. Conservative main text (includes table bodies/captions): {:,} words".format(
            report.conservative_main_text_words
        )
    )
    print("7. Methods: {:,} words".format(report.methods_words))
    print("8. Figure legends:")
    for legend in report.figure_legends:
        image = " [{}]".format(legend.image) if legend.image else ""
        print(
            "   Figure {} ({}): {:,} words{}".format(
                legend.number, legend.label, legend.words, image
            )
        )
    print(
        "9. Supplementary Information: {:,} words".format(
            report.supplementary_information_words
        )
    )
    print("10. Title: {:,} words".format(report.title_words))
    print("    {}".format(report.title))
    print(
        "11. Main display items: {} ({} figures + {} tables)".format(
            report.main_display_items, report.main_figures, report.main_tables
        )
    )
    print()
    print("Diagnostics:")
    for key, value in report.diagnostics.items():
        print("   {}: {:,}".format(key.replace("_", " "), value))
    print()
    print(
        "Policy: TeXcount prose words; headings, math, algorithms, bibliography, "
        "figure legends, and back matter are excluded from the main-text totals."
    )
    print(
        "The journal count is Introduction + Results + Discussion + Responsible "
        "use/limitations. The conservative count additionally includes table "
        "bodies and captions."
    )


def _quality_failures(report: ManuscriptReport) -> List[str]:
    failures = []
    if report.abstract_words > 200:
        failures.append("abstract exceeds 200 words")
    if report.title_words > 20:
        failures.append("title exceeds 20 words")
    if report.journal_main_text_words > 4500:
        failures.append("journal-defined main text exceeds 4,500 words")
    if report.main_display_items > 8:
        failures.append("main figures plus tables exceed 8")
    return failures


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--supplementary", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if an absolute journal limit is exceeded",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args.main.resolve(), args.supplementary.resolve())
    except (OSError, RuntimeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        _display_report(report)

    failures = _quality_failures(report) if args.check else []
    if failures:
        for failure in failures:
            print("FAIL: {}".format(failure), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
