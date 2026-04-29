"""Generate a PDF from a progress note markdown file."""

import markdown
import sys
from pathlib import Path

default_md = Path(__file__).parent / "progress_2026-04-21.md"
md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_md
if not md_path.is_absolute():
    md_path = Path(__file__).parent / md_path
pdf_path = md_path.with_suffix(".pdf")

md_text = md_path.read_text(encoding="utf-8")

html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    max-width: 800px;
    margin: 40px auto;
    color: #1a1a1a;
  }}
  h1 {{ font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 6px; margin-top: 0; }}
  h2 {{ font-size: 14pt; border-bottom: 1px solid #aaa; padding-bottom: 4px; margin-top: 28px; color: #222; }}
  h3 {{ font-size: 12pt; margin-top: 20px; color: #333; }}
  h4 {{ font-size: 11pt; margin-top: 14px; color: #444; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 10pt;
  }}
  th, td {{
    border: 1px solid #ccc;
    padding: 6px 10px;
    text-align: left;
  }}
  th {{ background-color: #f0f0f0; font-weight: bold; }}
  tr:nth-child(even) {{ background-color: #fafafa; }}
  code {{
    background: #f4f4f4;
    padding: 1px 4px;
    border-radius: 3px;
    font-family: "Consolas", monospace;
    font-size: 10pt;
  }}
  pre {{
    background: #f4f4f4;
    padding: 12px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 9.5pt;
  }}
  a {{ color: #0366d6; text-decoration: none; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 24px 0; }}
  li {{ margin: 3px 0; }}
  blockquote {{
    border-left: 3px solid #ccc;
    margin: 8px 0;
    padding-left: 14px;
    color: #555;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

# Try weasyprint first, fall back to reportlab-based HTML conversion
try:
    import weasyprint
    weasyprint.HTML(string=html).write_pdf(str(pdf_path))
    print(f"PDF saved via weasyprint: {pdf_path}")
except ImportError:
    # Use xhtml2pdf (pure Python)
    try:
        from xhtml2pdf import pisa
        with open(pdf_path, "wb") as f:
            pisa.CreatePDF(html.encode("utf-8"), dest=f)
        print(f"PDF saved via xhtml2pdf: {pdf_path}")
    except ImportError:
        # Last resort: use reportlab directly with plain text rendering
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib import colors
        import re

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
        )

        styles = getSampleStyleSheet()
        style_h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=8, textColor=colors.HexColor("#1a1a1a"))
        style_h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=18, spaceAfter=6, textColor=colors.HexColor("#222222"))
        style_h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceBefore=12, spaceAfter=4, textColor=colors.HexColor("#333333"))
        style_body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=15, spaceAfter=4)
        style_bullet = ParagraphStyle("Bullet", parent=styles["Normal"], fontSize=10, leading=14, leftIndent=16, spaceAfter=2)
        style_code = ParagraphStyle("Code", parent=styles["Code"], fontSize=8.5, leading=12, backColor=colors.HexColor("#f4f4f4"), leftIndent=12)

        story = []
        lines = md_text.split("\n")
        i = 0

        def strip_md_inline(text):
            """Strip inline markdown for reportlab."""
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
            text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
            text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
            text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
            text = text.replace("&", "&amp;").replace("<b>", "<b>").replace("</b>", "</b>")
            return text

        while i < len(lines):
            line = lines[i]

            # Headings
            if line.startswith("# "):
                story.append(Paragraph(line[2:], style_h1))
                story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#333333"), spaceAfter=6))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], style_h2))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa"), spaceAfter=4))
            elif line.startswith("### "):
                story.append(Paragraph(line[4:], style_h3))

            # Horizontal rule
            elif line.strip() == "---":
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd"), spaceBefore=8, spaceAfter=8))

            # Table detection
            elif line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
                table_lines = [line]
                j = i + 1
                while j < len(lines) and lines[j].startswith("|"):
                    table_lines.append(lines[j])
                    j += 1
                i = j - 1

                # Parse table
                rows = []
                for tl in table_lines:
                    if re.match(r"\|[-| :]+\|", tl):
                        continue
                    cells = [c.strip() for c in tl.strip("|").split("|")]
                    rows.append(cells)

                if rows:
                    col_count = max(len(r) for r in rows)
                    table_data = []
                    for row in rows:
                        while len(row) < col_count:
                            row.append("")
                        table_data.append([Paragraph(strip_md_inline(c), ParagraphStyle("TC", fontSize=9, leading=12)) for c in row])

                    tbl = Table(table_data, repeatRows=1)
                    tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ]))
                    story.append(tbl)
                    story.append(Spacer(1, 8))

            # Bullet points
            elif line.startswith("- ") or line.startswith("* "):
                content = line[2:]
                content = re.sub(r"\[x\]", "✓", content)
                content = re.sub(r"\[ \]", "☐", content)
                story.append(Paragraph("• " + strip_md_inline(content), style_bullet))

            # Code block
            elif line.startswith("```"):
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if code_lines:
                    story.append(Paragraph("<br/>".join(code_lines), style_code))
                    story.append(Spacer(1, 4))

            # Regular paragraph
            elif line.strip():
                story.append(Paragraph(strip_md_inline(line), style_body))

            # Empty line
            else:
                story.append(Spacer(1, 4))

            i += 1

        doc.build(story)
        print(f"PDF saved via reportlab: {pdf_path}")
