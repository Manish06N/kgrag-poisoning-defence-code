"""Minimal Markdown -> HTML for the paper draft (headings, pipe tables, lists, **bold**, *italic*, `code`, [[TODO/VERIFY]] markers).
Used only to print a PDF preview with headless Edge when no LaTeX compiler is installed; the authoritative typeset version is paper/main.tex.

  python -I scripts/md_to_html.py paper/draft.md paper/draft_preview.html
"""
import html
import re
import sys

AUTHORS = """<p class="authors">Manish Nandish<sup>1,2</sup>, Rajiv Misra<sup>1</sup>, Midhunchakkaravarthy Janarthanan<sup>2</sup></p>
<p class="affil"><sup>1</sup>Department of Computer Science and Engineering, Indian Institute of Technology Patna, Patna, Bihar, India<br>
manish_25s21res58@iitp.ac.in, rajivm@iitp.ac.in<br>
<sup>2</sup>Lincoln University College, Malaysia<br>midhun@lincoln.edu.my</p>"""

CSS = """@page { size: A4; margin: 22mm 20mm; }
body { font-family: 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.38; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 6px; }
h2 { font-size: 12.5pt; margin: 16px 0 4px; } h3 { font-size: 11pt; margin: 12px 0 3px; }
.authors { text-align: center; font-size: 11.5pt; margin: 4px 0 2px; } .affil { text-align: center; font-size: 9.5pt; font-style: italic; margin: 0 0 10px; }
p { margin: 4px 0; text-align: justify; } blockquote { border-left: 3px solid #999; margin: 8px 0; padding: 2px 10px; background: #f4f4f4; font-size: 9.5pt; }
table { border-collapse: collapse; margin: 8px auto; font-size: 9pt; } th, td { border-top: 1px solid #444; border-bottom: 1px solid #444; padding: 2px 6px; }
th { background: #eee; } code { font-family: Consolas, monospace; font-size: 9pt; } li { margin: 2px 0; }
.todo { color: #b00020; font-weight: bold; } .verify { color: #0b3d91; font-weight: bold; }"""


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"\[\[(TODO)(.*?)\]\]", r'<span class="todo">[\1\2]</span>', text, flags=re.S)
    text = re.sub(r"\[\[(VERIFY)(.*?)\]\]", r'<span class="verify">[\1\2]</span>', text, flags=re.S)
    return text


def convert(md: str) -> str:
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
        elif ln.startswith("# "):
            out.append(f"<h1>{inline(ln[2:])}</h1>" + AUTHORS)
            i += 1
        elif ln.startswith("### "):
            out.append(f"<h3>{inline(ln[4:])}</h3>")
            i += 1
        elif ln.startswith("## "):
            out.append(f"<h2>{inline(ln[3:])}</h2>")
            i += 1
        elif ln.startswith(">"):
            block = []
            while i < len(lines) and lines[i].startswith(">"):
                block.append(inline(lines[i].lstrip("> ")))
                i += 1
            out.append("<blockquote>" + "<br>".join(block) + "</blockquote>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            head, body = rows[0], rows[1:]
            out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>" +
                       "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</table>")
        elif re.match(r"(\d+\.|-) ", ln):
            tag = "ol" if ln[0].isdigit() else "ul"
            items = []
            while i < len(lines) and re.match(r"(\d+\.|-) ", lines[i]):
                items.append(inline(re.sub(r"^(\d+\.|-) ", "", lines[i].rstrip())))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{x}</li>" for x in items) + f"</{tag}>")
        else:
            para = [ln]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"(#|>|\||\d+\. |- )", lines[i]):
                para.append(lines[i].rstrip())
                i += 1
            out.append("<p>" + inline(" ".join(para)) + "</p>")
    return "\n".join(out)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    body = convert(open(src, encoding="utf8").read())
    open(dst, "w", encoding="utf8").write(
        f'<!doctype html><html><head><meta charset="utf-8"><title>KG-RAG poisoning defence (draft)</title><style>{CSS}</style></head><body>{body}</body></html>')
    print("wrote", dst)


if __name__ == "__main__":
    main()
