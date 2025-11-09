"""
Utility to convert Markdown documents into formatted DOCX files.

Supports IEEE-style two-column papers and single-column resumes by toggling
command-line arguments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


def add_heading_paragraph(
    document: Document, text: str, italic: bool = False, font_size: float = 10
) -> None:
    """
    Add a heading-styled paragraph with configurable styling.
    """
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(font_size)
    if italic:
        run.italic = True


def add_formatted_paragraph(document: Document, raw: str) -> None:
    """
    Parse simple bold (**text**) and italic (_text_) markers within a line.
    """
    paragraph = document.add_paragraph()
    pos = 0
    while pos < len(raw):
        if raw.startswith("**", pos):
            end = raw.find("**", pos + 2)
            if end == -1:
                end = len(raw)
            run = paragraph.add_run(raw[pos + 2 : end])
            run.bold = True
            pos = end + 2
        elif raw.startswith("_", pos):
            end = raw.find("_", pos + 1)
            if end == -1:
                end = len(raw)
            run = paragraph.add_run(raw[pos + 1 : end])
            run.italic = True
            pos = end + 1
        else:
            run = paragraph.add_run(raw[pos])
            pos += 1


def add_table(document: Document, rows: list[str]) -> None:
    """
    Convert markdown-style table rows into a DOCX table.
    """
    header = [cell.strip() for cell in rows[0].strip("|").split("|")]
    body_rows = [r for r in rows[2:]]
    table = document.add_table(rows=1 + len(body_rows), cols=len(header))
    table.style = "Table Grid"
    for idx, text in enumerate(header):
        table.cell(0, idx).text = text
    for row_idx, row in enumerate(body_rows, start=1):
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        for col_idx, cell_text in enumerate(cells):
            table.cell(row_idx, col_idx).text = cell_text


def convert_markdown_to_docx(
    source: Path,
    target: Path,
    *,
    font_name: str = "Times New Roman",
    base_font_size: float = 10,
    title_font_size: float = 26,
    two_column: bool = False,
) -> None:
    """
    Convert a Markdown file to DOCX with configurable layout and typography.
    """
    if not source.exists():
        raise FileNotFoundError(f"Source markdown not found: {source}")

    lines = source.read_text(encoding="utf-8").splitlines()
    document = Document()

    normal_style = document.styles["Normal"]
    normal_style.font.name = font_name
    normal_style.font.size = Pt(base_font_size)

    for section in document.sections:
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        if two_column:
            cols = section._sectPr.xpath("./w:cols")
            if cols:
                cols[0].set(qn("w:num"), "2")
                cols[0].set(qn("w:space"), "720")

    i = 0
    in_code_block = False
    code_buffer: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("%"):
            i += 1
            continue

        if stripped.startswith("# "):
            title = stripped[2:].strip()
            title_paragraph = document.add_paragraph()
            title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = title_paragraph.add_run(title)
            run.font.size = Pt(title_font_size)
            run.bold = True
            run.font.name = font_name
            i += 1
            continue

        if stripped.startswith("## "):
            add_heading_paragraph(
                document, stripped[3:].strip(), font_size=base_font_size
            )
            i += 1
            continue

        if stripped.startswith("### "):
            add_heading_paragraph(
                document, stripped[4:].strip(), italic=True, font_size=base_font_size
            )
            i += 1
            continue

        if stripped.startswith("- "):
            document.add_paragraph(stripped[2:].strip(), style="List Bullet")
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            add_table(document, table_lines)
            continue

        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_buffer = []
            else:
                in_code_block = False
                if code_buffer:
                    for item in code_buffer:
                        placeholder = document.add_paragraph(f"[Figure: {item}]")
                        placeholder.alignment = WD_ALIGN_PARAGRAPH.CENTER
                code_buffer = []
            i += 1
            continue

        if in_code_block:
            code_buffer.append(stripped)
            i += 1
            continue

        if stripped.startswith("\\["):
            equation_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("\\]"):
                equation_lines.append(lines[i].strip())
                i += 1
            i += 1  # Skip closing \]
            equation_paragraph = document.add_paragraph(" ".join(equation_lines))
            equation_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in equation_paragraph.runs:
                run.italic = True
            continue

        if stripped.startswith("[") and stripped[1:2].isdigit():
            document.add_paragraph(stripped)
            i += 1
            continue

        if stripped.startswith("**") or "_" in stripped:
            add_formatted_paragraph(document, line)
            i += 1
            continue

        document.add_paragraph(line.strip())
        i += 1

    target.parent.mkdir(parents=True, exist_ok=True)
    document.save(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Markdown to DOCX.")
    parser.add_argument("source", type=Path, help="Path to the Markdown file.")
    parser.add_argument(
        "--target",
        type=Path,
        help="Optional output DOCX path (defaults to same name).",
    )
    parser.add_argument(
        "--two-column",
        action="store_true",
        help="Use a two-column layout (IEEE-style).",
    )
    parser.add_argument(
        "--title-size",
        type=float,
        default=26.0,
        help="Font size for top-level title (default: 26).",
    )
    parser.add_argument(
        "--font-name",
        default="Times New Roman",
        help="Base font name (default: Times New Roman).",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=10.0,
        help="Base body font size (default: 10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = (
        args.target if args.target is not None else args.source.with_suffix(".docx")
    )
    convert_markdown_to_docx(
        args.source,
        output_path,
        font_name=args.font_name,
        base_font_size=args.font_size,
        title_font_size=args.title_size,
        two_column=args.two_column,
    )

