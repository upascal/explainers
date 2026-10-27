"""Assemble the GPT-2 embeddings explainer from src/ into site/ (dev, external assets) and dist/ (self-contained).

    python gpt2-embeddings/build.py

src/index.md        prose. Headings "## 4 Title" get id="s4" and the number span; the TOC
                    is generated from them at the <!-- toc --> marker. Widgets are pulled in
                    at <!-- module: NAME --> markers from src/modules/NAME.html.
src/template.html   page shell with {{title}} {{css}} {{content}} {{js}}
src/gpt2data.js     real GPT-2 data, written by precompute/precompute.py
src/model.js        GPT-2's tokenizer and the toy one-layer GPT
src/app.js          the interactive code; src/style.css
"""
import pathlib, re
from markdown_it import MarkdownIt

ROOT = pathlib.Path(__file__).parent
SRC, SITE, DIST = ROOT / "src", ROOT / "site", ROOT / "dist"
HEAD = re.compile(r"<h([23])>(\d+(?:\.\d+)*)\s+(.*?)</h\1>")
SLOT = re.compile(r"<!-- module: (\w+) -->")
JS = ("gpt2data.js", "model.js", "app.js")


def render_content():
    html = MarkdownIt("commonmark").enable("table").render((SRC / "index.md").read_text())
    toc = []

    def head(m):
        lvl, num, text = m.groups()
        hid = "s" + num.replace(".", "-")
        toc.append(f'<a href="#{hid}">{num if "." in num else num + "."} {text}</a>')
        return f'<h{lvl} id="{hid}"><span class="num">{num}</span>{text}</h{lvl}>'

    html = HEAD.sub(head, html)
    html = html.replace("<!-- toc -->", '<nav class="toc">\n  ' + "\n  ".join(toc) + "\n</nav>", 1)
    used = set()

    def slot(m):
        used.add(m[1])
        return (SRC / "modules" / f"{m[1]}.html").read_text().rstrip()

    html = SLOT.sub(slot, html)
    unused = {p.stem for p in (SRC / "modules").glob("*.html")} - used
    assert not unused, f"modules never placed in index.md: {sorted(unused)}"
    title = re.sub(r"<.*?>", "", re.search(r"<h1>(.*?)</h1>", html).group(1))
    return html, title


def page(content, title, inline):
    t = (SRC / "template.html").read_text()
    css, js = (SRC / "style.css").read_text(), "\n".join((SRC / f).read_text() for f in JS)
    t = t.replace("{{title}}", title).replace("{{content}}", content)
    t = t.replace("{{css}}", f"<style>\n{css}</style>" if inline else '<link rel="stylesheet" href="style.css">')
    t = t.replace("{{js}}", f"<script>\n{js}</script>" if inline else "\n".join(f'<script src="{f}"></script>' for f in JS))
    return t


def write(path, text):
    path.write_text(text)
    print(f"wrote {path.relative_to(ROOT)} ({len(text) / 1e6:.2f} MB)")


content, title = render_content()
SITE.mkdir(exist_ok=True); DIST.mkdir(exist_ok=True)
write(SITE / "index.html", page(content, title, inline=False))
for f in ("style.css", *JS):
    (SITE / f).write_text((SRC / f).read_text())
write(DIST / "index.html", page(content, title, inline=True))
