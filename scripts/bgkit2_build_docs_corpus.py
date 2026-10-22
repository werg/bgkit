#!/usr/bin/env python
"""Build a library / framework documentation-page corpus for a searchable docs tool.

Output: ``<out-dir>/pages.parquet`` with columns

  id (str)       ``<library>/<relative path>`` plus ``#partN`` for split pages
  library (str)  python | numpy | pandas | requests | pytest | click | pydantic |
                 torch | transformers | git | bash | coreutils
  version (str)  documentation version (pinned tag / archive version)
  title (str)    page title (``"... (part N)"`` for continuation parts)
  url (str)      canonical URL of the rendered page
  text (str)     plain text — headings kept as lines, code blocks kept verbatim
                 inside ``` fences, no HTML / RST / Markdown inline markup
  n_chars (int)  len(text)
  split (str)    "eval" for 2% of base page ids (sha1), else "train"

and ``<out-dir>/manifest.json`` (per-library counts, sources, versions, fetch
commands).  Pages are 300 .. ~30,000 characters; longer pages are split at
section headings into consecutive parts.

Sources (all fetched into ``<out-dir>/raw/``; each is skipped when present):

  python        docs.python.org plain-text archive (library/, reference/,
                howto/, tutorial/)
  numpy, pandas, requests, pytest, click, pydantic, torch, transformers, git
                shallow + blob-less sparse git clones of the docs directory at
                the pinned tag (a few MB each; torch/transformers ~30 MB)
  bash, coreutils
                the GNU single-page HTML manuals (2 requests)

numpy / pandas / torch additionally get one page per public API object,
generated from the docstrings of the venv-installed package (which is pinned
to the same version as the cloned docs).  ``.. autofunction::`` /
``[[autodoc]]`` / ``::: module`` markers inside the docs are expanded the same
way.

Converters: no docutils in the venv, so RST is converted by a hand-written
state machine (sections, directives, literal/doctest blocks, inline roles);
Markdown (MyST / mkdocs / HF flavours) and AsciiDoc (git) by regex passes;
HTML by BeautifulSoup.

Resumable: raw fetches are cached, per-library page lists are written to
``<out-dir>/work/<library>.jsonl`` and reused; the final parquet is never
overwritten unless ``--overwrite`` is given.

Example:
  .venv/bin/python scripts/bgkit2_build_docs_corpus.py
  .venv/bin/python scripts/bgkit2_build_docs_corpus.py --rebuild numpy,pandas --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

OUT_DIR = "/home/werg/bgkit-data-nvme/bgkit2/docs"
MIN_CHARS = 300
MAX_CHARS = 30_000
EVAL_PCT = 2
USER_AGENT = "bgkit2-docs-corpus/0.1 (research corpus builder; polite, single-threaded)"
REQUEST_SLEEP_S = 1.0

# --------------------------------------------------------------------------- sources
GIT = "https://github.com/{repo}"

SOURCES: dict[str, dict] = {
    "python": {
        "version": "3.12",
        "kind": "tarball",
        "url": "https://docs.python.org/3.12/archives/python-3.12-docs-text.tar.bz2",
        "subdirs": ["library", "reference", "howto", "tutorial"],
        "url_template": "https://docs.python.org/3.12/{rel}.html",
        "format": "pytext",
    },
    "numpy": {
        "version": "2.4.4",
        "kind": "git",
        "repo": "numpy/numpy",
        "tag": "v2.4.4",
        "sparse": ["doc/source"],
        "doc_root": "doc/source",
        "include": ["user", "reference"],
        "exclude": ["reference/random/bit_generators", "reference/c-api"],
        "url_template": "https://numpy.org/doc/2.4/{rel}.html",
        "format": "rst",
        "api": "numpy",
    },
    "pandas": {
        "version": "3.0.2",
        "kind": "git",
        "repo": "pandas-dev/pandas",
        "tag": "v3.0.2",
        "sparse": ["doc/source"],
        "doc_root": "doc/source",
        "include": ["user_guide", "getting_started", "development", "reference"],
        "exclude": ["reference/api"],
        "url_template": "https://pandas.pydata.org/docs/{rel}.html",
        "format": "rst",
        "api": "pandas",
    },
    "requests": {
        "version": "2.33.1",
        "kind": "git",
        "repo": "psf/requests",
        "tag": "v2.33.1",
        "sparse": ["docs"],
        "doc_root": "docs",
        "include": [""],
        "exclude": ["_static", "_templates", "_themes", "dev/authors"],
        "url_template": "https://requests.readthedocs.io/en/latest/{rel}/",
        "format": "rst",
    },
    "pytest": {
        "version": "9.0.3",
        "kind": "git",
        "repo": "pytest-dev/pytest",
        "tag": "9.0.3",
        "sparse": ["doc/en"],
        "doc_root": "doc/en",
        "include": [""],
        "exclude": ["announce", "changelog", "_templates", "example/assertion", "proposals"],
        "url_template": "https://docs.pytest.org/en/stable/{rel}.html",
        "format": "rst",
    },
    "click": {
        "version": "8.3.2",
        "kind": "git",
        "repo": "pallets/click",
        "tag": "8.3.2",
        "sparse": ["docs"],
        "doc_root": "docs",
        "include": [""],
        "exclude": ["changes", "_static", "license"],
        "url_template": "https://click.palletsprojects.com/en/stable/{rel}/",
        "format": "auto",
    },
    "pydantic": {
        "version": "2.13.0",
        "kind": "git",
        "repo": "pydantic/pydantic",
        "tag": "v2.13.0",
        "sparse": ["docs"],
        "doc_root": "docs",
        "include": [""],
        "exclude": ["plugins", "theme", "extra", "img", "logos", "sponsor_logos", "badge",
                    "changelog", "index", "help_with_pydantic", "pydantic_people",
                    "internals", "contributing"],
        "url_template": "https://docs.pydantic.dev/latest/{rel}/",
        "format": "md",
    },
    "torch": {
        "version": "2.11.0",
        "kind": "git",
        "repo": "pytorch/pytorch",
        "tag": "v2.11.0",
        "sparse": ["docs/source"],
        "doc_root": "docs/source",
        "include": [""],
        "exclude": ["_static", "_templates", "scripts", "community", "cpp_index",
                    "conf", "deploy", "index"],
        "url_template": "https://docs.pytorch.org/docs/stable/{rel}.html",
        "format": "auto",
        "api": "torch",
    },
    "transformers": {
        "version": "5.5.4",
        "kind": "git",
        "repo": "huggingface/transformers",
        "tag": "v5.5.4",
        "sparse": ["docs/source/en"],
        "doc_root": "docs/source/en",
        "include": [""],
        "exclude": ["_toctree", "_config", "_redirects", "index", "internal"],
        "url_template": "https://huggingface.co/docs/transformers/v5.5.4/en/{rel}",
        "format": "md",
        "autodoc_module": "transformers",
        # model pages list ``Model - forward`` for hundreds of near-identical templated
        # forward docstrings: keep the class docstring, members as signature lines only
        "autodoc_member_docs": False,
        "autodoc_budget": 10_000,
    },
    "git": {
        "version": "2.55.0",
        "kind": "git",
        "repo": "git/git",
        "tag": "v2.55.0",
        "sparse": ["Documentation"],
        "doc_root": "Documentation",
        "include": [""],
        "exclude": [],
        "only_glob": "git*.adoc",
        "url_template": "https://git-scm.com/docs/{rel}",
        "format": "adoc",
    },
    "bash": {
        "version": "5.2",
        "kind": "html",
        "url": "https://www.gnu.org/software/bash/manual/bash.html",
        "url_template": "https://www.gnu.org/software/bash/manual/bash.html#{anchor}",
        "format": "html",
    },
    "coreutils": {
        "version": "9.x",
        "kind": "html",
        "url": "https://www.gnu.org/software/coreutils/manual/coreutils.html",
        "url_template": "https://www.gnu.org/software/coreutils/manual/coreutils.html#{anchor}",
        "format": "html",
    },
}


# --------------------------------------------------------------------------- fetch
def _ipv4_only() -> None:
    # gnu.org publishes an AAAA record this host cannot reach (SYN-SENT forever); urllib3
    # tries IPv6 first and burns the whole connect timeout, so resolve IPv4 only.
    import urllib3.util.connection as _c

    _c.HAS_IPV6 = False


def _http_get(url: str, dest: Path) -> None:
    import requests

    _ipv4_only()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True,
                      timeout=(20, 120)) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    time.sleep(REQUEST_SLEEP_S)


def _robots_allows(url: str) -> bool:
    from urllib import robotparser
    from urllib.parse import urlsplit

    import requests

    parts = urlsplit(url)
    robots = f"{parts.scheme}://{parts.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    _ipv4_only()
    try:
        r = requests.get(robots, headers={"User-Agent": USER_AGENT}, timeout=(20, 30))
        time.sleep(REQUEST_SLEEP_S)
        if r.status_code != 200:
            return True
        rp.parse(r.text.splitlines())
    except Exception:
        return True
    return rp.can_fetch("*", url)


def fetch(lib: str, spec: dict, raw_dir: Path) -> tuple[Path, list[str]]:
    """Fetch the raw docs for one library (skipped when the .done marker exists)."""
    dest = raw_dir / f"{lib}_{spec['version']}"
    cmds: list[str] = []
    if spec["kind"] == "tarball":
        cmds = [f"curl -A '{USER_AGENT}' -o {dest}.tar.bz2 {spec['url']}", "tar -xjf (subdirs)"]
    elif spec["kind"] == "git":
        cmds = [
            f"git clone --depth 1 --filter=blob:none --sparse --branch {spec['tag']} "
            f"{GIT.format(repo=spec['repo'])} {dest}",
            f"git -C {dest} sparse-checkout set {' '.join(spec['sparse'])}",
        ]
    else:
        cmds = [f"curl -A '{USER_AGENT}' -o {dest}/page.html {spec['url']}"]
    marker = dest / ".done"
    if marker.exists():
        return dest, cmds
    if dest.exists():
        shutil.rmtree(dest)
    print(f"[fetch] {lib}: {cmds[0]}", flush=True)
    if spec["kind"] == "tarball":
        if not _robots_allows(spec["url"]):
            sys.exit(f"robots.txt disallows {spec['url']}")
        tb = raw_dir / f"{lib}_{spec['version']}.tar.bz2"
        if not tb.exists():
            _http_get(spec["url"], tb)
        dest.mkdir(parents=True)
        with tarfile.open(tb, "r:bz2") as tf:
            for m in tf.getmembers():
                parts = Path(m.name).parts
                if len(parts) >= 3 and parts[1] in spec["subdirs"] and m.isfile():
                    data = tf.extractfile(m).read()
                    p = dest / Path(*parts[1:])
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(data)
    elif spec["kind"] == "git":
        subprocess.run(
            ["git", "-c", "advice.detachedHead=false", "clone", "-q", "--depth", "1",
             "--filter=blob:none", "--sparse", "--branch", spec["tag"],
             GIT.format(repo=spec["repo"]), str(dest)],
            check=True,
        )
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "-q", *spec["sparse"]],
                       check=True)
        time.sleep(REQUEST_SLEEP_S)
    else:
        if not _robots_allows(spec["url"]):
            sys.exit(f"robots.txt disallows {spec['url']}")
        _http_get(spec["url"], dest / "page.html")
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return dest, cmds


# --------------------------------------------------------------------------- blocks
# A converted document is a list of blocks: ("h", level, text) | ("p", text) | ("code", text).
Block = tuple


def H(level: int, text: str) -> Block:  # noqa: N802 - block constructors
    return ("h", level, text.strip())


def P(text: str) -> Block:  # noqa: N802
    return ("p", text)


def CODE(text: str, lang: str = "") -> Block:  # noqa: N802
    return ("code", lang, text.rstrip("\n"))


_HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(\s[^<>]*)?/?>")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def _clean_para(text: str) -> str:
    text = "\n".join(x.rstrip() for x in text.split("\n"))
    return text.strip("\n")


def render_blocks(blocks: list[Block]) -> str:
    out: list[str] = []
    for b in blocks:
        if b[0] == "h":
            out.append(b[2])
        elif b[0] == "p":
            t = _clean_para(b[1])
            if t.strip():
                out.append(t)
        else:
            out.append(f"```{b[1]}\n{b[2]}\n```")
    text = "\n\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dedent(lines: list[str]) -> list[str]:
    ind = [len(x) - len(x.lstrip()) for x in lines if x.strip()]
    if not ind:
        return [x.strip() for x in lines]
    m = min(ind)
    return [x[m:] if x.strip() else "" for x in lines]


# --------------------------------------------------------------------------- RST
_ROLE_RE = re.compile(r":[a-zA-Z0-9_.+:-]+:`([^`]*)`")
_ROLE_TARGET_RE = re.compile(r"^(.*?)\s*<[^<>]*>$")
_LIT_RE = re.compile(r"``(.+?)``", re.S)
_LINK_RE = re.compile(r"`([^`<]*?)\s*<[^`>]*>`__?")
_REF_RE = re.compile(r"`([^`]+)`__?")
_INTERP_RE = re.compile(r"(?<![`\w])`([^`\n]+)`(?![`\w])")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_EM_RE = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_SUB_RE = re.compile(r"\|([A-Za-z0-9_ .+-]+)\|")
_FOOT_RE = re.compile(r"\[(#?[A-Za-z0-9_]*|\*)\]_")
_WORDREF_RE = re.compile(r"\b([A-Za-z][\w.-]*)_(?=[\s,.;:)]|$)")


def _role_sub(m: re.Match) -> str:
    inner = m.group(1)
    t = _ROLE_TARGET_RE.match(inner)
    if t and t.group(1):
        return t.group(1)
    if inner.startswith("~"):
        return inner[1:].split(".")[-1]
    if inner.startswith("!"):
        return inner[1:]
    return inner


def rst_inline(line: str) -> str:
    line = _ROLE_RE.sub(_role_sub, line)
    line = _LIT_RE.sub(r"\1", line)
    line = _LINK_RE.sub(r"\1", line)
    line = _REF_RE.sub(r"\1", line)
    line = _INTERP_RE.sub(r"\1", line)
    line = _BOLD_RE.sub(r"\1", line)
    line = _EM_RE.sub(r"\1", line)
    line = _SUB_RE.sub(r"\1", line)
    line = _FOOT_RE.sub(lambda m: f"[{m.group(1) or '*'}]", line)
    line = _WORDREF_RE.sub(r"\1", line)
    line = line.replace("\\ ", "").replace("\\*", "*").replace("\\`", "`").replace("\\_", "_")
    return line


_UNDERLINE_RE = re.compile(r"^([=\-~^\"'`*+#.:_])\1{2,}\s*$")
_DIRECTIVE_RE = re.compile(r"^(\s*)\.\.\s+([a-zA-Z0-9_:.+-]+)::(.*)$")
_COMMENT_RE = re.compile(r"^(\s*)\.\.(\s.*|)$")
_OPTION_RE = re.compile(r"^\s*:([a-zA-Z0-9_-]+):(.*)$")

ADMONITIONS = {
    "note": "Note", "warning": "Warning", "tip": "Tip", "hint": "Hint", "important": "Important",
    "caution": "Caution", "danger": "Danger", "attention": "Attention", "error": "Error",
    "seealso": "See also", "admonition": "", "versionadded": "Added in version",
    "versionchanged": "Changed in version", "deprecated": "Deprecated since version",
    "versionremoved": "Removed in version", "todo": "Todo", "rubric": "", "topic": "",
    "sidebar": "", "deprecated-removed": "Deprecated since version",
}
CODE_DIRECTIVES = {"code-block", "code", "sourcecode", "literal", "doctest", "testcode",
                   "testoutput", "click:example", "ipython", "parsed-literal", "jupyter-execute"}
DROP_DIRECTIVES = {"toctree", "contents", "image", "figure", "include", "index", "raw",
                   "highlight", "autosummary", "only", "click:run", "literalinclude", "csv-table",
                   "list-table", "plot", "graphviz", "mermaid", "meta", "sectionauthor",
                   "moduleauthor", "codeauthor", "tabularcolumns", "glossary", "productionlist",
                   "customcarditem", "customcardstart", "customcardend", "customcalloutitem",
                   "displayitem", "galleryitem", "thumbnail", "youtube", "gallery", "role",
                   "default-role", "cssclass", "rst-class", "footer", "header", "date",
                   "release", "pypa-tabs", "hlist", "table", "tabs", "group-tab", "tab",
                   "code-tab", "opengraph", "vimeo"}
AUTODOC_DIRECTIVES = {"autofunction", "autoclass", "autoexception", "autodata", "automethod",
                      "autoattribute", "automodule", "autoproperty", "autodecorator"}
SIGNATURE_DIRECTIVES = {"function", "class", "method", "exception", "data", "attribute",
                        "module", "decorator", "property", "staticmethod", "classmethod",
                        "cfunction", "ctype", "cvar", "cmacro", "c:function", "c:type",
                        "py:function", "py:class", "py:method", "py:data", "py:attribute",
                        "option", "envvar", "confval", "describe", "object", "cmdoption",
                        "program", "fixture", "opcode", "pdbcommand", "monitoring-event",
                        "audit-event", "decoratormethod", "coroutinefunction",
                        "coroutinemethod", "awaitablefunction", "awaitablemethod",
                        "abstractmethod", "js:function", "js:class", "type", "attributetable"}


class RstConverter:
    def __init__(self, autodoc, current_module: str | None = None):
        self.autodoc = autodoc  # callable(dotted, options, module) -> list[Block]
        self.module = current_module
        self.levels: dict[str, int] = {}

    def _level(self, ch: str) -> int:
        if ch not in self.levels:
            self.levels[ch] = len(self.levels) + 1
        return self.levels[ch]

    def convert(self, text: str) -> list[Block]:
        lines = text.replace("\r\n", "\n").replace("\t", "    ").split("\n")
        return self._convert_lines(lines)

    def _convert_lines(self, lines: list[str]) -> list[Block]:
        blocks: list[Block] = []
        i, n = 0, len(lines)
        para: list[str] = []

        def flush():
            nonlocal para
            if para:
                joined = rst_inline("\n".join(para))
                if joined.strip():
                    blocks.append(P(joined))
                para = []

        while i < n:
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                flush()
                i += 1
                continue
            # section heading: (overline?) title / underline
            if (i + 1 < n and _UNDERLINE_RE.match(lines[i + 1])
                    and len(lines[i + 1].strip()) >= min(len(stripped), 3)
                    and not line.startswith(" ") and not _UNDERLINE_RE.match(line)):
                flush()
                blocks.append(H(self._level(lines[i + 1].strip()[0]), rst_inline(stripped)))
                i += 2
                continue
            if _UNDERLINE_RE.match(line) and i + 2 < n and _UNDERLINE_RE.match(lines[i + 2]) \
                    and lines[i + 1].strip():
                flush()
                blocks.append(H(self._level(line.strip()[0]), rst_inline(lines[i + 1].strip())))
                i += 3
                continue
            if _UNDERLINE_RE.match(line) and len(stripped) >= 4:
                i += 1  # stray transition / underline
                continue
            m = _DIRECTIVE_RE.match(line)
            if m:
                flush()
                indent = len(m.group(1))
                name = m.group(2).lower()
                arg = m.group(3).strip()
                j = i + 1
                body: list[str] = []
                while j < n and (not lines[j].strip() or
                                 len(lines[j]) - len(lines[j].lstrip()) > indent):
                    body.append(lines[j])
                    j += 1
                while body and not body[-1].strip():
                    body.pop()
                opts: dict[str, str] = {}
                k = 0
                while k < len(body) and _OPTION_RE.match(body[k]):
                    om = _OPTION_RE.match(body[k])
                    opts[om.group(1)] = om.group(2).strip()
                    k += 1
                if arg and not opts and body and body[0].strip() and \
                        name not in CODE_DIRECTIVES and name not in AUTODOC_DIRECTIVES:
                    # directive argument continued on the following indented lines
                    while k < len(body) and body[k].strip():
                        arg += " " + body[k].strip()
                        k += 1
                while k < len(body) and not body[k].strip():
                    k += 1
                body = _dedent(body[k:])
                blocks.extend(self._directive(name, arg, opts, body))
                i = j
                continue
            if _COMMENT_RE.match(line) and not stripped.startswith(".. |"):
                # comment / hyperlink target / substitution: drop with its indented body
                flush()
                indent = len(line) - len(line.lstrip())
                j = i + 1
                while j < n and (not lines[j].strip() or
                                 len(lines[j]) - len(lines[j].lstrip()) > indent):
                    j += 1
                i = j
                continue
            if stripped.startswith(".. |"):
                i += 1
                continue
            if stripped.startswith(">>> "):
                flush()
                indent = len(line) - len(line.lstrip())
                j = i
                code: list[str] = []
                while j < n and lines[j].strip() and \
                        len(lines[j]) - len(lines[j].lstrip()) >= indent:
                    code.append(lines[j][indent:])
                    j += 1
                blocks.append(CODE("\n".join(code), "python"))
                i = j
                continue
            # paragraph line; a trailing "::" introduces a literal block
            if stripped.endswith("::"):
                head = stripped[:-2].rstrip()
                if head:
                    para.append(line[: len(line) - len(line.lstrip())] + head +
                                ("" if stripped.endswith(" ::") or head.endswith(":") else ":"))
                flush()
                indent = len(line) - len(line.lstrip())
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                code = []
                while j < n and (not lines[j].strip() or
                                 len(lines[j]) - len(lines[j].lstrip()) > indent):
                    code.append(lines[j])
                    j += 1
                while code and not code[-1].strip():
                    code.pop()
                if code:
                    blocks.append(CODE("\n".join(_dedent(code))))
                i = j
                continue
            para.append(line)
            i += 1
        flush()
        return blocks

    def _directive(self, name: str, arg: str, opts: dict, body: list[str]) -> list[Block]:
        if name in ("currentmodule", "module", "py:module", "py:currentmodule"):
            self.module = arg or self.module
            if name in ("module", "py:module") and body:
                return self._convert_lines(body)
            return []
        if name in CODE_DIRECTIVES:
            lang = arg.split()[0] if arg and name in ("code-block", "code", "sourcecode") else ""
            if name == "click:example":
                lang = "python"
            return [CODE("\n".join(body), lang)] if body else []
        if name == "math":
            return [P(arg + ("\n" + "\n".join(body) if body else ""))]
        if name in DROP_DIRECTIVES:
            return []
        if name in AUTODOC_DIRECTIVES:
            if name == "automodule" and arg:
                self.module = arg.strip()
            return self.autodoc(arg, opts, self.module, name) if arg else []
        if name in SIGNATURE_DIRECTIVES:
            out: list[Block] = []
            if arg:
                out.append(P(rst_inline(arg)))
            if body:
                out.extend(self._convert_lines(body))
            return out
        if name in ADMONITIONS:
            label = ADMONITIONS[name]
            text = rst_inline(arg)
            head = f"{label}: {text}" if label and text else (label + ":" if label else text)
            inner = self._convert_lines(body) if body else []
            return ([P(head)] if head else []) + inner
        if name == "epigraph" or name == "pull-quote" or name == "highlights":
            return self._convert_lines(body)
        # unknown directive: keep its body text (most sphinx extensions wrap content)
        return self._convert_lines(body) if body else []


# --------------------------------------------------------------------------- Markdown
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^()]*\))*\)")
_MD_HFREF_RE = re.compile(r"\[`~?([^`\]]+)`\]")
_MD_ALERT_RE = re.compile(r"^>\s*\[!([A-Z]+)\]\s*$", re.M)
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_REFLINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_MD_ROLE_RE = re.compile(r"\{[a-zA-Z0-9_:.+-]+\}`([^`]*)`")
_MD_CODE_RE = re.compile(r"`([^`\n]+)`")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_MD_EM_RE = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_MD_ATTR_RE = re.compile(r"\s*\{[ .#][^}]*\}\s*$")
_MD_TARGET_RE = re.compile(r"^\([A-Za-z0-9_.:-]+\)=\s*$")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,}|:{3,})\s*(\{?[^\s`~{}]*\}?)\s*(.*)$")
_MD_ADMON_RE = re.compile(r'^(!!!|\?\?\?\+?)\s+([a-zA-Z_-]+)(?:\s+"([^"]*)")?\s*$')
_MD_TAB_RE = re.compile(r'^===\s+"([^"]*)"\s*$')
_MD_SNIPPET_RE = re.compile(r"^--8<--")
_MD_AUTODOC_RE = re.compile(r"^\[\[autodoc\]\]\s+([A-Za-z0-9_.]+)\s*(.*)$")
_MD_MKAUTODOC_RE = re.compile(r"^:::\s*([A-Za-z0-9_.]+)\s*$")
_MD_HTML_BLOCK_TAGS = {"tip", "hfoptions", "hfoption", "div", "details", "summary",
                       "frameworkcontent", "pt", "tf", "jax", "video", "iframe", "img", "br",
                       "p", "a", "span", "b",
                       "sup", "sub", "figure", "figcaption", "source", "center", "small", "table",
                       "tr", "td", "th", "thead", "tbody", "ul", "ol", "li", "h1", "h2", "h3",
                       "code", "pre", "em", "strong", "kbd", "audio", "hr", "picture", "u", "i",
                       "font", "script", "style", "blockquote"}


def md_inline(line: str) -> str:
    line = _MD_IMG_RE.sub("", line)
    line = _MD_HFREF_RE.sub(r"\1", line)
    line = _MD_ROLE_RE.sub(_role_sub, line)
    line = _ROLE_RE.sub(_role_sub, line)
    line = _MD_ALERT_RE.sub(lambda m: m.group(1).capitalize() + ":", line)
    line = re.sub(r"^>\s?", "", line, flags=re.M)  # blockquote prefixes
    line = _MD_LINK_RE.sub(r"\1", line)
    line = _MD_REFLINK_RE.sub(r"\1", line)
    line = _LIT_RE.sub(r"\1", line)
    line = _MD_CODE_RE.sub(r"\1", line)
    line = _MD_BOLD_RE.sub(lambda m: m.group(1) or m.group(2) or "", line)
    line = _MD_EM_RE.sub(r"\1", line)
    line = _MD_ATTR_RE.sub("", line)
    line = line.replace("\\_", "_").replace("\\*", "*").replace("\\<", "<").replace("\\>", ">")
    line = _HTML_TAG_RE.sub("", line)
    return html.unescape(line)


class MdConverter:
    def __init__(self, autodoc, current_module: str | None = None):
        self.autodoc = autodoc
        self.module = current_module
        self.rst = RstConverter(autodoc, current_module)

    def convert(self, text: str) -> list[Block]:
        text = text.replace("\r\n", "\n")
        text = _HTML_COMMENT_RE.sub("", text)
        lines = text.split("\n")
        if lines and lines[0].strip() == "---":  # front matter
            try:
                end = lines.index("---", 1)
                lines = lines[end + 1:]
            except ValueError:
                pass
        return self._convert_lines(lines)

    def _convert_lines(self, lines: list[str]) -> list[Block]:
        blocks: list[Block] = []
        para: list[str] = []
        i, n = 0, len(lines)

        def flush():
            nonlocal para
            if para:
                joined = md_inline("\n".join(para))
                if joined.strip():
                    blocks.append(P(joined))
                para = []

        while i < n:
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                flush()
                i += 1
                continue
            fm = _MD_FENCE_RE.match(line)
            if fm and (fm.group(2)[0] != ":" or fm.group(3).startswith("{")):
                flush()
                indent, fence, info = fm.group(1), fm.group(2), fm.group(3)
                j = i + 1
                body: list[str] = []
                closer = re.compile(rf"^{re.escape(fence[0])}{{{len(fence)},}}\s*$")
                while j < n and not closer.match(lines[j].strip()):
                    body.append(lines[j][len(indent):] if lines[j].startswith(indent)
                                else lines[j])
                    j += 1
                if j >= n and len(fence) > 3:
                    i += 1  # unterminated long fence (upstream typo): treat as plain text
                    continue
                i = j + 1
                if info.startswith("{"):
                    blocks.extend(self._myst_directive(info.strip("{}").lower(),
                                                       fm.group(4).strip(), body))
                else:
                    lang = info.split()[0] if info else ""
                    if lang in ("mermaid", "toctree", "{toctree}"):
                        continue
                    blocks.append(CODE("\n".join(body), lang))
                continue
            if _MD_TARGET_RE.match(stripped) or _MD_SNIPPET_RE.match(stripped) or \
                    stripped == "[[open-in-colab]]":
                i += 1
                continue
            hm = _MD_HEADING_RE.match(line)
            if hm:
                flush()
                blocks.append(H(len(hm.group(1)), md_inline(hm.group(2))))
                i += 1
                continue
            if i + 1 < n and re.match(r"^(=+|-+)\s*$", lines[i + 1]) and not line.startswith(" ") \
                    and not stripped.startswith(("|", "-", "*", "+")) and len(stripped) < 120:
                flush()
                blocks.append(H(1 if lines[i + 1].startswith("=") else 2, md_inline(stripped)))
                i += 2
                continue
            am = _MD_AUTODOC_RE.match(stripped)
            if am:
                flush()
                members = [x.strip() for x in am.group(2).split("-") if x.strip()]
                j = i + 1
                while j < n and re.match(r"^\s+-\s+\S+", lines[j]):
                    members.append(lines[j].strip()[1:].strip())
                    j += 1
                blocks.extend(self.autodoc(am.group(1), {"members": ",".join(members)},
                                           self.module, "autoclass"))
                i = j
                continue
            mm = _MD_MKAUTODOC_RE.match(stripped)
            if mm:
                flush()
                j = i + 1
                opt_lines: list[str] = []
                while j < n and (lines[j].startswith("    ") or not lines[j].strip()):
                    if not lines[j].strip() and (j + 1 >= n or not lines[j + 1].startswith("    ")):
                        break
                    opt_lines.append(lines[j])
                    j += 1
                members = re.findall(r"^\s+-\s+(\S+)", "\n".join(opt_lines), re.M)
                blocks.extend(self.autodoc(mm.group(1), {"members": ",".join(members)},
                                           self.module, "automodule"))
                i = j
                continue
            adm = _MD_ADMON_RE.match(stripped)
            tab = _MD_TAB_RE.match(stripped)
            if adm or tab:
                flush()
                if adm:
                    title = adm.group(3)
                    head = adm.group(2).capitalize() + (f": {title}" if title else ":")
                else:
                    head = tab.group(1) + ":"
                j = i + 1
                body = []
                while j < n and (not lines[j].strip() or lines[j].startswith("    ")):
                    body.append(lines[j][4:] if lines[j].startswith("    ") else "")
                    j += 1
                blocks.append(P(head))
                blocks.extend(self._convert_lines(body))
                i = j
                continue
            # html-ish block tags (HF <Tip>, <hfoptions>, ...): keep semantic label, drop tag
            tm = re.match(r"^<(/?)([A-Za-z][A-Za-z0-9]*)([^>]*)>\s*(.*)$", stripped)
            if tm and tm.group(2).lower() in _MD_HTML_BLOCK_TAGS:
                tag = tm.group(2).lower()
                rest = tm.group(4)
                if not tm.group(1):
                    if tag == "tip":
                        flush()
                        blocks.append(P("Warning:" if "warning" in tm.group(3) else "Tip:"))
                    elif tag == "hfoption":
                        idm = re.search(r'id="([^"]*)"', tm.group(3))
                        flush()
                        if idm:
                            blocks.append(P(f"[{idm.group(1)}]"))
                if rest and not re.match(r"^</?[A-Za-z]", rest):
                    para.append(_HTML_TAG_RE.sub("", rest))
                i += 1
                continue
            para.append(line)
            i += 1
        flush()
        return blocks

    def _myst_directive(self, name: str, arg: str, body: list[str]) -> list[Block]:
        name = name.strip()
        if name in ("eval-rst",):
            return self.rst._convert_lines(body)
        if name in ("currentmodule", "module"):
            self.module = arg or self.module
            self.rst.module = self.module
            return []
        opts: dict[str, str] = {}
        k = 0
        while k < len(body) and _OPTION_RE.match(body[k]):
            om = _OPTION_RE.match(body[k])
            opts[om.group(1)] = om.group(2).strip()
            k += 1
        while k < len(body) and not body[k].strip():
            k += 1
        body = body[k:]
        if name in CODE_DIRECTIVES:
            return [CODE("\n".join(body), arg.split()[0] if arg else "")] if body else []
        if name in AUTODOC_DIRECTIVES:
            return self.autodoc(arg, opts, self.module, name) if arg else []
        if name in DROP_DIRECTIVES:
            return []
        if name in SIGNATURE_DIRECTIVES:
            out = [P(md_inline(arg))] if arg else []
            return out + self._convert_lines(body)
        if name in ADMONITIONS:
            label = ADMONITIONS[name]
            text = md_inline(arg)
            head = f"{label}: {text}" if label and text else (label + ":" if label else text)
            inner = self._convert_lines(body)
            return ([P(head)] if head else []) + inner
        if name in ("tab-set", "tab-item", "grid", "grid-item-card", "card", "dropdown",
                    "collapse", "div", "container", "tabs", "tab", "group-tab", "code-tab"):
            return ([P(md_inline(arg) + ":")] if arg and name != "grid" else []) + \
                self._convert_lines(body)
        return self._convert_lines(body)


# --------------------------------------------------------------------------- AsciiDoc (git)
_ADOC_INCLUDE_RE = re.compile(r"^include::(\S+?)\[\]\s*$")
_ADOC_LINKGIT_RE = re.compile(r"linkgit:([A-Za-z0-9_.-]+)\[(\d+)\]")
_ADOC_LINK_RE = re.compile(r"link:(\S+?)\[([^\]]*)\]")
_ADOC_MACRO_RE = re.compile(r"\b(?:ifdef|ifndef|endif)::[^\n]*")


def adoc_to_blocks(text: str, root: Path, depth: int = 0) -> list[Block]:
    lines = text.replace("\r\n", "\n").split("\n")
    blocks: list[Block] = []
    para: list[str] = []
    i, n = 0, len(lines)

    def inline(s: str) -> str:
        s = _ADOC_LINKGIT_RE.sub(r"\1(\2)", s)
        s = _ADOC_LINK_RE.sub(lambda m: m.group(2) or m.group(1), s)
        s = re.sub(r"(?<![\w`])`([^`\n]+)`(?![\w`])", r"\1", s)
        s = re.sub(r"(?<![\w'])'([^'\n]+)'(?![\w'])", r"\1", s)
        s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\1", s)
        s = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"\1", s)
        s = s.replace("{tilde}", "~").replace("{plus}", "+").replace("{litdd}", "--")
        s = re.sub(r"\+\+([^+\n]+)\+\+", r"\1", s)
        s = re.sub(r"\[verse\]|\[literal\]|\[synopsis\]|^\[\[[^\]]*\]\]$", "", s)
        return s

    def flush():
        nonlocal para
        if para:
            joined = "\n".join(inline(x) for x in para)
            if joined.strip():
                blocks.append(P(joined))
            para = []

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or _ADOC_MACRO_RE.match(stripped) or stripped.startswith("//") or \
                (stripped.startswith(":") and stripped.endswith(":")) or \
                re.match(r"^\[\[[^\]]*\]\]$", stripped) or re.match(r"^:[a-z_-]+:", stripped):
            if not stripped:
                flush()
            i += 1
            continue
        inc = _ADOC_INCLUDE_RE.match(stripped)
        if inc:
            flush()
            p = root / inc.group(1)
            if depth < 3 and p.exists():
                blocks.extend(adoc_to_blocks(p.read_text("utf-8", "replace"), root, depth + 1))
            i += 1
            continue
        if re.match(r"^(-{4,}|\.{4,}|\+{4,})\s*$", stripped):
            flush()
            fence = stripped[0]
            j = i + 1
            body = []
            while j < n and not re.match(rf"^{re.escape(fence)}{{4,}}\s*$", lines[j].strip()):
                body.append(lines[j])
                j += 1
            if body:
                blocks.append(CODE("\n".join(body)))
            i = j + 1
            continue
        if re.match(r"^={2,}\s*$", stripped) or re.match(r"^\[[a-zA-Z_,=\" -]*\]$", stripped):
            i += 1
            continue
        hm = re.match(r"^(=+)\s+(.*?)\s*=*\s*$", line)
        if hm and not line.startswith(" "):
            flush()
            blocks.append(H(len(hm.group(1)), inline(hm.group(2))))
            i += 1
            continue
        if i + 1 < n and re.match(r"^([=\-~^]){3,}\s*$", lines[i + 1]) and \
                abs(len(lines[i + 1].rstrip()) - len(stripped)) <= 2 and not line.startswith(" "):
            flush()
            ch = lines[i + 1][0]
            blocks.append(H({"=": 1, "-": 2, "~": 3, "^": 4}[ch], inline(stripped)))
            i += 2
            continue
        if stripped.endswith("::") and not line.startswith(" "):
            # labeled list item: "--option::" followed by indented description
            flush()
            para.append(stripped[:-2] + ":")
            i += 1
            continue
        para.append(line)
        i += 1
    flush()
    return blocks


# --------------------------------------------------------------------------- HTML (GNU manuals)
def html_to_pages(raw: str, spec: dict, lib: str) -> list[dict]:
    """One page per section (h2/h3); h4+ nested inside."""
    from bs4 import BeautifulSoup, NavigableString

    soup = BeautifulSoup(raw, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer"]):
        t.decompose()
    for t in soup.select("div.nav-panel, div.header, div.contents, div.shortcontents, "
                         "a.copiable-link, div.element-contents, div.contents-heading"):
        t.decompose()
    body = soup.body or soup
    pages: list[dict] = []
    cur: dict | None = None
    pending_anchor: str | None = None

    def start(level: int, title: str, anchor: str | None):
        nonlocal cur
        cur = {"title": title, "anchor": anchor, "blocks": [H(level, title)]}
        pages.append(cur)

    for el in body.descendants:
        if isinstance(el, NavigableString):
            continue
        name = el.name
        if name in ("span", "a") and el.get("id") and not el.get_text(strip=True):
            pending_anchor = el["id"]
        if name in ("h1", "h2", "h3", "h4", "h5"):
            level = int(name[1])
            title = re.sub(r"\s+", " ", el.get_text("", strip=True)).replace("¶", "").strip()
            title = re.sub(r"^([A-Z]|\d+)(\.\d+)*\s+", "", title)
            anchor = el.get("id") or pending_anchor
            pending_anchor = None
            if level <= 3 or cur is None:
                start(level, title, anchor)
            else:
                cur["blocks"].append(H(level, title))
        elif cur is not None and name == "pre":
            if el.find_parent("pre") is None:
                cur["blocks"].append(CODE(el.get_text()))
        elif cur is not None and name in ("p", "dt", "dd", "li", "th", "td", "blockquote"):
            if el.find_parent(["pre", "li", "dd", "td", "th", "blockquote"]) is not None and \
                    name in ("p", "li", "dd"):
                # nested: let the outermost container emit
                if el.find_parent(["li", "dd", "td", "th", "blockquote"]).name in ("li", "dd") \
                        and name == "p":
                    continue
                if name in ("li", "dd") and el.find_parent(["td", "th"]) is None:
                    pass
                else:
                    continue
            if el.find("pre") is not None:
                for pre in el.find_all("pre"):
                    pre.replace_with(soup.new_string("\n" + pre.get_text() + "\n"))
            txt = el.get_text("", strip=False)
            txt = re.sub(r"\s+", " ", txt).strip()
            if txt and not re.match(r"^(Next|Previous|Up):", txt) and "[ Contents ]" not in txt:
                cur["blocks"].append(P(("- " if name == "li" else "") + txt))
    out = []
    skip = re.compile(r"^(Short )?Table of Contents$|Index$|^Index\b")
    for pg in pages:
        if skip.search(pg["title"]):
            continue
        anchor = pg["anchor"] or re.sub(r"[^A-Za-z0-9]+", "-", pg["title"])
        out.append({
            "rel": anchor,
            "title": pg["title"],
            "url": spec["url_template"].format(anchor=anchor),
            "blocks": pg["blocks"],
        })
    return out


# --------------------------------------------------------------------------- python text archive
def pytext_to_blocks(text: str) -> list[Block]:
    lines = text.replace("\r\n", "\n").split("\n")
    blocks: list[Block] = []
    para: list[str] = []
    levels: dict[str, int] = {}
    i, n = 0, len(lines)

    def flush():
        nonlocal para
        if para:
            blocks.append(P("\n".join(_pytext_inline(x) for x in para)))
            para = []

    while i < n:
        line = lines[i]
        if not line.strip():
            flush()
            i += 1
            continue
        if re.match(r"^([=\-~^\"*#])\1{3,}\s*$", line) and (i == 0 or not lines[i - 1].strip()):
            i += 1  # transition line
            continue
        if (i + 1 < n and re.match(r"^([=\-~^\"*#]|\')\1{2,}\s*$", lines[i + 1])
                and not line.startswith(" ") and len(lines[i + 1].rstrip()) >= len(line.rstrip())
                and (i == 0 or not lines[i - 1].strip())):
            flush()
            ch = lines[i + 1][0]
            levels.setdefault(ch, len(levels) + 1)
            blocks.append(H(levels[ch], _pytext_inline(line.strip())))
            i += 2
            continue
        para.append(line)
        i += 1
    flush()
    return blocks


def _pytext_inline(line: str) -> str:
    """The Sphinx text builder keeps ``**strong**`` / ``*emphasis*`` markers; drop them."""
    if line.startswith("   ") or line.startswith(">>>"):
        return line  # indented = literal block / table in the text build
    line = _BOLD_RE.sub(r"\1", line)
    return _EM_RE.sub(r"\1", line)


# --------------------------------------------------------------------------- autodoc / API pages
_OBJ_CACHE: dict[str, object] = {}


def _import_object(dotted: str, module: str | None):
    cands = [dotted]
    if module and not dotted.startswith(module + "."):
        cands.insert(0, f"{module}.{dotted}")
    for c in cands:
        if c in _OBJ_CACHE:
            return _OBJ_CACHE[c], c
        parts = c.split(".")
        for i in range(len(parts), 0, -1):
            modname = ".".join(parts[:i])
            try:
                obj = importlib.import_module(modname)
            except Exception:
                continue
            ok = True
            for attr in parts[i:]:
                try:
                    obj = getattr(obj, attr)
                except Exception:
                    ok = False
                    break
            if ok:
                _OBJ_CACHE[c] = obj
                return obj, c
    return None, None


def _own_doc(obj) -> str:
    try:
        doc = getattr(obj, "__doc__", None)
    except Exception:
        return ""
    if not isinstance(doc, str):
        return ""
    if inspect.isclass(obj):
        for base in obj.__mro__[1:]:
            if getattr(base, "__doc__", None) == doc:
                return ""
    return inspect.cleandoc(doc)


def _signature(obj, name: str) -> str:
    try:
        sig = str(inspect.signature(obj)).replace("<no value>", "NoValue")
        sig = re.sub(r"<([^<>]*)>", r"\1", sig)
        if len(sig) > 600:
            sig = sig[:600] + "...)"
        return name + sig
    except Exception:
        return name


_RST_DIRECTIVE_HINT_RE = re.compile(r"^\s*\.\. [a-z]+::", re.M)


def _truncate_blocks(blocks: list[Block], budget: int) -> list[Block]:
    """Keep whole blocks up to ``budget`` characters, then mark the cut."""
    out: list[Block] = []
    size = 0
    for b in blocks:
        size += len(b[-1])
        if size > budget and out:
            out.append(P("[...]"))
            break
        out.append(b)
    return out


class Autodoc:
    """Expand autodoc-style markers from the venv-installed package."""

    def __init__(self, fmt: str, default_module: str | None = None, member_docs: bool = True,
                 budget: int = 40_000):
        self.fmt = fmt  # "rst" | "md" — docstring dialect
        self.default_module = default_module
        self.member_docs = member_docs  # False: listed members get a signature line only
        self.budget = budget
        self.misses = Counter()
        self.hits = 0

    def doc_blocks(self, doc: str) -> list[Block]:
        if not doc:
            return []
        looks_rst = _RST_DIRECTIVE_HINT_RE.search(doc) and "```" not in doc
        if not looks_rst and (self.fmt == "md" or "```" in doc or "[[autodoc]]" in doc
                              or "<Tip" in doc):
            return MdConverter(self.__call__)._convert_lines(doc.split("\n"))
        return RstConverter(self.__call__)._convert_lines(doc.split("\n"))

    def render(self, obj, name: str, members: list[str] | None, with_members: bool,
               budget: int | None = None) -> list[Block]:
        budget = self.budget if budget is None else budget
        head = _signature(obj, name) if callable(obj) and not inspect.ismodule(obj) else name
        blocks: list[Block] = [P(head)]
        blocks.extend(_truncate_blocks(self.doc_blocks(_own_doc(obj)), budget))
        size = sum(len(b[-1]) for b in blocks)
        names: list[str] = []
        if members:
            names = members
        elif with_members and (inspect.isclass(obj) or inspect.ismodule(obj)):
            pub = getattr(obj, "__all__", None) if inspect.ismodule(obj) else None
            if pub is None:
                pub = [k for k in vars(obj) if not k.startswith("_")]
            names = list(pub)[:80]
        for m in names:
            if size > budget:
                break
            try:
                sub = getattr(obj, m, None)
            except Exception:
                continue
            if sub is None or inspect.ismodule(sub):
                continue
            if inspect.ismodule(obj) and getattr(sub, "__module__", obj.__name__) != obj.__name__ \
                    and not inspect.isclass(sub):
                continue
            d = _own_doc(sub)
            if not d:
                continue
            blocks.append(P(_signature(sub, f"{name}.{m}") if callable(sub) else f"{name}.{m}"))
            if not self.member_docs:
                continue
            sub_blocks = self.doc_blocks(d)
            blocks.extend(sub_blocks)
            size += sum(len(b[-1]) for b in sub_blocks)
        return blocks

    def __call__(self, dotted: str, opts: dict, module: str | None, directive: str) -> list[Block]:
        target = dotted.split("(")[0].strip()
        obj, resolved = _import_object(target, module or self.default_module)
        if obj is None:
            self.misses[target] += 1
            return [P(dotted)]
        self.hits += 1
        members = [x.strip() for x in opts.get("members", "").split(",") if x.strip()]
        with_members = "members" in opts and not members and directive != "automodule"
        return self.render(obj, resolved, members or None, with_members)


def api_pages(lib: str, spec: dict, autodoc: Autodoc) -> list[dict]:
    """One page per public API object from the installed package's docstrings."""
    ver = spec["version"]
    entries: list[tuple[str, object, str]] = []  # (qualname, obj, url)
    seen: set[int] = set()

    def add(prefix: str, obj, url_fn, names=None, pred=None):
        names = names if names is not None else [k for k in dir(obj) if not k.startswith("_")]
        for k in names:
            try:
                sub = getattr(obj, k)
            except Exception:
                continue
            if inspect.ismodule(sub) or (pred and not pred(sub)):
                continue
            if not (callable(sub) or inspect.isclass(sub) or isinstance(sub, property)
                    or type(sub).__name__ in ("getset_descriptor", "method_descriptor",
                                              "wrapper_descriptor", "CachedAccessor",
                                              "_tuplegetter")):
                continue
            key = id(sub)
            if key in seen:
                continue
            seen.add(key)
            entries.append((f"{prefix}.{k}", sub, url_fn(f"{prefix}.{k}")))

    if lib == "numpy":
        import numpy as np

        u = "https://numpy.org/doc/2.4/reference/generated/{q}.html"
        gen = u.format
        add("numpy", np, lambda q: gen(q=q), names=sorted(np.__all__))
        add("numpy.ndarray", np.ndarray, lambda q: gen(q=q))
        for sub in ("linalg", "fft", "random", "ma", "strings", "testing", "polynomial.polynomial"):
            mod = importlib.import_module(f"numpy.{sub}")
            add(f"numpy.{sub}", mod, lambda q: gen(q=q),
                names=sorted(getattr(mod, "__all__", []) or
                             [k for k in dir(mod) if not k.startswith("_")]))
        add("numpy.random.Generator", np.random.Generator, lambda q: gen(q=q))
    elif lib == "pandas":
        import pandas as pd

        u = "https://pandas.pydata.org/docs/reference/api/{q}.html"
        gen = u.format
        add("pandas", pd, lambda q: gen(q=q), names=sorted(pd.__all__))
        for cls_name in ("DataFrame", "Series", "Index", "MultiIndex", "DatetimeIndex",
                         "TimedeltaIndex", "PeriodIndex", "CategoricalIndex", "IntervalIndex",
                         "Timestamp", "Timedelta", "Period", "Interval", "Categorical",
                         "ExcelWriter", "HDFStore"):
            add(f"pandas.{cls_name}", getattr(pd, cls_name), lambda q: gen(q=q))
        from pandas.core import groupby, resample, window
        from pandas.core.arrays.categorical import CategoricalAccessor
        from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
        from pandas.core.strings.accessor import StringMethods

        add("pandas.Series.str", StringMethods, lambda q: gen(q=q))
        add("pandas.Series.dt", CombinedDatetimelikeProperties, lambda q: gen(q=q))
        add("pandas.Series.cat", CategoricalAccessor, lambda q: gen(q=q))
        for q, obj in (("pandas.core.groupby.DataFrameGroupBy", groupby.DataFrameGroupBy),
                       ("pandas.core.groupby.SeriesGroupBy", groupby.SeriesGroupBy),
                       ("pandas.core.window.rolling.Rolling", window.Rolling),
                       ("pandas.core.window.expanding.Expanding", window.Expanding),
                       ("pandas.core.window.ewm.ExponentialMovingWindow",
                        window.ExponentialMovingWindow),
                       ("pandas.core.resample.Resampler", resample.Resampler)):
            add(q, obj, lambda q: gen(q=q))
        add("pandas.api.types", pd.api.types, lambda q: gen(q=q))
        add("pandas.plotting", pd.plotting, lambda q: gen(q=q))
    elif lib == "torch":
        import torch

        u = "https://docs.pytorch.org/docs/stable/generated/{q}.html"
        gen = u.format
        add("torch", torch, lambda q: gen(q=q), names=sorted(torch.__all__))
        add("torch.Tensor", torch.Tensor, lambda q: gen(q=q))
        add("torch.nn", torch.nn, lambda q: gen(q=q),
            pred=lambda o: inspect.isclass(o) and issubclass(o, torch.nn.Module))
        add("torch.nn.functional", torch.nn.functional, lambda q: gen(q=q),
            pred=inspect.isroutine)
        add("torch.optim", torch.optim, lambda q: gen(q=q),
            pred=lambda o: inspect.isclass(o) and issubclass(o, torch.optim.Optimizer))
        add("torch.optim.lr_scheduler", torch.optim.lr_scheduler, lambda q: gen(q=q),
            pred=inspect.isclass)
        for sub in ("linalg", "fft", "special", "sparse", "autograd", "cuda", "distributed",
                    "utils.data", "utils.checkpoint", "func", "distributions", "jit", "amp",
                    "backends.cudnn", "onnx", "profiler", "nn.init", "nn.utils",
                    "nn.utils.rnn", "nn.attention", "export", "compiler", "testing"):
            try:
                mod = importlib.import_module(f"torch.{sub}")
            except Exception:
                continue
            page = {"utils.data": "data.html#{q}", "utils.checkpoint": "checkpoint.html#{q}",
                    "autograd": "autograd.html#{q}", "cuda": "cuda.html#{q}",
                    "distributed": "distributed.html#{q}", "func": "func.api.html#{q}",
                    "distributions": "distributions.html#{q}", "jit": "jit.html#{q}",
                    "amp": "amp.html#{q}", "backends.cudnn": "backends.html#{q}",
                    "onnx": "onnx.html#{q}", "profiler": "profiler.html#{q}",
                    "nn.init": "nn.init.html#{q}", "export": "export.html#{q}",
                    "compiler": "torch.compiler_api.html#{q}",
                    "testing": "testing.html#{q}"}.get(sub, "generated/{q}.html")
            tmpl = "https://docs.pytorch.org/docs/stable/" + page
            names = sorted(getattr(mod, "__all__", None) or
                           [k for k in dir(mod) if not k.startswith("_")])
            add(f"torch.{sub}", mod, lambda q, t=tmpl: t.format(q=q), names=names,
                pred=lambda o: (inspect.isroutine(o) or inspect.isclass(o)))
    pages = []
    for q, obj, url in entries:
        doc = _own_doc(obj)
        if len(doc) < 120:
            continue
        blocks = [H(1, q), *autodoc.render(obj, q, None, False)]
        pages.append({"rel": f"api/{q}", "title": q, "url": url, "blocks": blocks,
                      "version": ver})
    return pages


# --------------------------------------------------------------------------- paging
def split_into_parts(blocks: list[Block]) -> list[list[Block]]:
    """Pack section-wise into parts of <= MAX_CHARS, splitting only at headings when possible."""
    rendered = [(b, len(render_blocks([b])) + 2) for b in blocks]
    sections: list[list[tuple[Block, int]]] = []
    for b, ln in rendered:
        if b[0] == "h" or not sections:
            sections.append([])
        sections[-1].append((b, ln))

    def chunk_big(sec):
        chunks, cur, size = [], [], 0
        for b, ln in sec:
            if ln > MAX_CHARS:  # giant single block: split by lines
                text = b[-1]
                step = MAX_CHARS - 200
                for k in range(0, len(text), step):
                    piece = text[k:k + step]
                    nb = CODE(piece, b[1]) if b[0] == "code" else P(piece)
                    if cur:
                        chunks.append(cur)
                        cur, size = [], 0
                    chunks.append([(nb, len(piece) + 10)])
                continue
            if size + ln > MAX_CHARS and cur:
                chunks.append(cur)
                cur, size = [], 0
            cur.append((b, ln))
            size += ln
        if cur:
            chunks.append(cur)
        return chunks

    parts: list[list[tuple[Block, int]]] = []
    cur: list[tuple[Block, int]] = []
    size = 0
    for sec in sections:
        sec_len = sum(ln for _, ln in sec)
        if sec_len > MAX_CHARS:
            if cur:
                parts.append(cur)
                cur, size = [], 0
            parts.extend(chunk_big(sec))
            continue
        if size + sec_len > MAX_CHARS and cur:
            parts.append(cur)
            cur, size = [], 0
        cur.extend(sec)
        size += sec_len
    if cur:
        parts.append(cur)
    # merge a tiny trailing part into its predecessor
    if len(parts) >= 2 and sum(ln for _, ln in parts[-1]) < MIN_CHARS:
        parts[-2].extend(parts.pop())
    return [[b for b, _ in p] for p in parts]


def page_title(blocks: list[Block], fallback: str) -> str:
    for b in blocks:
        if b[0] == "h":
            t = re.sub(r'^"([^"]+)"\s+---\s+', r"\1 — ", b[2])
            return t.strip() or fallback
    return fallback


def split_for(base_id: str) -> str:
    h = int(hashlib.sha1(base_id.encode()).hexdigest(), 16)
    return "eval" if h % 100 < EVAL_PCT else "train"


def emit_pages(lib: str, version: str, rel: str, title: str | None, url: str,
               blocks: list[Block]) -> list[dict]:
    rows = []
    if not blocks:
        return rows
    base_id = f"{lib}/{rel}"
    base_title = title or page_title(blocks, rel.rsplit("/", 1)[-1])
    parts = split_into_parts(blocks)
    for k, part in enumerate(parts, 1):
        text = render_blocks(part)
        if len(text) < MIN_CHARS:
            continue
        rows.append({
            "id": base_id + (f"#part{k}" if k > 1 else ""),
            "library": lib,
            "version": version,
            "title": base_title + (f" (part {k})" if k > 1 else ""),
            "url": url,
            "text": text,
            "n_chars": len(text),
            "split": split_for(base_id),
        })
    return rows


# --------------------------------------------------------------------------- per-library build
def _iter_doc_files(root: Path, spec: dict):
    exts = {"rst": (".rst",), "md": (".md",), "auto": (".rst", ".md"), "adoc": (".adoc",),
            "pytext": (".txt",)}[spec["format"]]
    glob = spec.get("only_glob")
    for inc in spec.get("include", spec.get("subdirs", [""])):
        base = root / inc if inc else root
        if not base.exists():
            continue
        for p in sorted(base.rglob(glob or "*")):
            if not p.is_file() or p.suffix not in exts:
                continue
            rel = p.relative_to(root).with_suffix("").as_posix()
            excl = spec.get("exclude", [])
            if any(rel == ex or rel.startswith(ex.rstrip("/") + "/") for ex in excl):
                continue
            if p.name.startswith(("_", ".")) and p.name not in ("_index.md",):
                continue
            yield p, rel


def build_library(lib: str, spec: dict, raw_dir: Path, no_autodoc: bool) -> tuple[list[dict], dict]:
    dest, cmds = fetch(lib, spec, raw_dir)
    rows: list[dict] = []
    info = {"version": spec["version"], "fetch_commands": cmds,
            "source": spec.get("url") or GIT.format(repo=spec["repo"]) + "@" + spec.get("tag", "")}
    t0 = time.time()
    fmt_doc = "md" if lib in ("pydantic", "transformers") else "rst"
    autodoc = Autodoc(fmt_doc, spec.get("autodoc_module"),
                      member_docs=spec.get("autodoc_member_docs", True),
                      budget=spec.get("autodoc_budget", 40_000))
    autodoc_fn = (lambda *a, **k: []) if no_autodoc else autodoc
    n_files = 0
    if spec["format"] == "html":
        for pg in html_to_pages((dest / "page.html").read_text("utf-8", "replace"), spec, lib):
            rows.extend(emit_pages(lib, spec["version"], pg["rel"], pg["title"], pg["url"],
                                   pg["blocks"]))
        n_files = 1
    else:
        root = dest / spec.get("doc_root", "")
        for path, rel in _iter_doc_files(root, spec):
            n_files += 1
            text = path.read_text("utf-8", "replace")
            if spec["format"] == "pytext":
                blocks = pytext_to_blocks(text)
            elif spec["format"] == "adoc":
                blocks = adoc_to_blocks(text, root)
            elif path.suffix == ".md":
                blocks = MdConverter(autodoc_fn).convert(text)
            else:
                blocks = RstConverter(autodoc_fn).convert(text)
            if lib == "git":
                rel = path.stem
                if blocks and blocks[0][0] == "h" and blocks[0][2].lower() == "name" and \
                        len(blocks) > 1 and blocks[1][0] == "p":
                    blocks = [H(1, blocks[1][1].split("\n")[0]), *blocks[2:]]
            url_rel = rel[:-len("/index")] if rel.endswith("/index") and rel != "index" else rel
            url = spec["url_template"].format(rel=url_rel)
            rows.extend(emit_pages(lib, spec["version"], rel, None, url, blocks))
        if spec.get("api") and not no_autodoc:
            api = api_pages(lib, spec, autodoc)
            info["api_objects"] = len(api)
            for pg in api:
                rows.extend(emit_pages(lib, pg["version"], pg["rel"], pg["title"], pg["url"],
                                       pg["blocks"]))
    info.update({"source_files": n_files, "pages": len(rows),
                 "chars": sum(r["n_chars"] for r in rows),
                 "autodoc_hits": autodoc.hits, "autodoc_misses": sum(autodoc.misses.values()),
                 "seconds": round(time.time() - t0, 1)})
    if autodoc.misses:
        info["autodoc_miss_examples"] = [k for k, _ in autodoc.misses.most_common(8)]
    return rows, info


# --------------------------------------------------------------------------- verify
_HTML_CHECK_RE = re.compile(
    r"</?(div|span|p|a|img|br|table|tr|td|th|ul|ol|li|h[1-6]|script|style|pre|code|em|strong|b|i)"
    r"(\s[^<>]*)?>", re.I)
_MARKUP_CHECK_RES = {
    "rst_role": re.compile(r":(func|class|meth|ref|mod|attr|exc|data|term|doc):`"),
    "rst_directive": re.compile(r"^\.\. [a-z-]+::", re.M),
    "md_link": re.compile(r"\[[^\]]+\]\(https?://"),
    "hf_autodoc": re.compile(r"\[\[autodoc\]\]"),
}


def _strip_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def verify(df) -> dict:
    import numpy as np

    rep: dict = {"n_pages": len(df), "n_chars": int(df.n_chars.sum())}
    prose = df.text.map(_strip_code)
    rep["pages_with_html_tags_outside_code"] = int(
        prose.map(lambda t: bool(_HTML_CHECK_RE.search(t))).sum())
    rep["html_tag_examples"] = [
        (r.id, _HTML_CHECK_RE.search(p).group(0))
        for r, p in zip(df.itertuples(), prose, strict=True) if _HTML_CHECK_RE.search(p)
    ][:10]
    for k, rx in _MARKUP_CHECK_RES.items():
        rep[f"pages_with_{k}"] = int(prose.map(lambda t, rx=rx: bool(rx.search(t))).sum())
    rep["pages_with_code_blocks"] = int(df.text.str.contains("```", regex=False).sum())
    rep["unbalanced_code_fences"] = int(df.text.map(lambda t: t.count("```") % 2 == 1).sum())
    rep["min_chars"] = int(df.n_chars.min())
    rep["max_chars"] = int(df.n_chars.max())
    rep["p50_chars"] = int(np.percentile(df.n_chars, 50))
    rep["p90_chars"] = int(np.percentile(df.n_chars, 90))
    rep["split_counts"] = {k: int(v) for k, v in df.split.value_counts().items()}
    rep["duplicate_ids"] = int(df.id.duplicated().sum())
    per = {}
    for lib, g in df.groupby("library"):
        per[lib] = {"pages": len(g), "chars": int(g.n_chars.sum()),
                    "p50": int(np.percentile(g.n_chars, 50)),
                    "p90": int(np.percentile(g.n_chars, 90)),
                    "parts": int(g.id.str.contains("#part").sum()),
                    "eval": int((g.split == "eval").sum())}
    rep["per_library"] = per
    return rep


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--libraries", default=",".join(SOURCES), help="comma-separated subset")
    ap.add_argument("--rebuild", nargs="?", const="all", default="",
                    help="ignore cached work/<lib>.jsonl: 'all' (default) or a comma list")
    ap.add_argument("--refetch", action="store_true", help="ignore cached raw fetches")
    ap.add_argument("--no-autodoc", action="store_true",
                    help="do not import packages to expand autodoc markers / API pages")
    ap.add_argument("--overwrite", action="store_true", help="overwrite an existing pages.parquet")
    ap.add_argument("--fetch-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir, work_dir = out_dir / "raw", out_dir / "work"
    raw_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "pages.parquet"
    if parquet.exists() and not args.overwrite and not args.fetch_only:
        sys.exit(f"refusing to overwrite existing {parquet} (pass --overwrite)")
    libs = [x.strip() for x in args.libraries.split(",") if x.strip()]
    unknown = [x for x in libs if x not in SOURCES]
    if unknown:
        sys.exit(f"unknown libraries: {unknown}; known: {list(SOURCES)}")

    manifest: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "script": "scripts/bgkit2_build_docs_corpus.py",
                      "min_chars": MIN_CHARS, "max_chars": MAX_CHARS, "eval_pct": EVAL_PCT,
                      "converters": {
                          "rst": "hand-written state machine (no docutils in venv)",
                          "md": "regex passes (MyST / mkdocs / HF flavours)",
                          "adoc": "regex passes", "html": "beautifulsoup4",
                          "pytext": "Sphinx text builder output, headings parsed"},
                      "libraries": {}}
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        with contextlib.suppress(Exception):
            manifest["libraries"] = json.loads(manifest_path.read_text())["libraries"]

    all_rows: list[dict] = []
    for lib in libs:
        spec = SOURCES[lib]
        if args.refetch:
            marker = raw_dir / f"{lib}_{spec['version']}" / ".done"
            if marker.exists():
                marker.unlink()
        if args.fetch_only:
            fetch(lib, spec, raw_dir)
            continue
        work = work_dir / f"{lib}.jsonl"
        rebuild = args.rebuild == "all" or lib in args.rebuild.split(",")
        if work.exists() and not rebuild:
            rows = [json.loads(x) for x in work.read_text("utf-8").splitlines()]
            print(f"[reuse] {lib}: {len(rows)} pages from {work}", flush=True)
        else:
            rows, info = build_library(lib, spec, raw_dir, args.no_autodoc)
            manifest["libraries"][lib] = info
            tmp = work.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, work)
            print(f"[build] {lib}: {json.dumps(info)}", flush=True)
        all_rows.extend(rows)
    if args.fetch_only:
        print("fetch-only done", flush=True)
        return

    import pandas as pd

    df = pd.DataFrame(all_rows, columns=["id", "library", "version", "title", "url", "text",
                                         "n_chars", "split"])
    df = df.drop_duplicates("id").reset_index(drop=True)
    df["n_chars"] = df["n_chars"].astype("int64")
    tmp = parquet.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, parquet)
    report = verify(df)
    manifest["verification"] = report
    manifest["output"] = str(parquet)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\nwrote {parquet}: {len(df)} pages, {df.n_chars.sum():,} chars")
    print(f"p50 {report['p50_chars']}  p90 {report['p90_chars']}  min {report['min_chars']}  "
          f"max {report['max_chars']}  split {report['split_counts']}")
    print(f"html tags outside code: {report['pages_with_html_tags_outside_code']} pages "
          f"{report['html_tag_examples'][:5]}")
    for k in _MARKUP_CHECK_RES:
        print(f"  residual {k}: {report[f'pages_with_{k}']} pages")
    print(f"pages with code blocks: {report['pages_with_code_blocks']}  "
          f"unbalanced fences: {report['unbalanced_code_fences']}")
    print(f"{'library':13s} {'pages':>7s} {'chars':>12s} {'p50':>7s} {'p90':>7s} "
          f"{'parts':>6s} {'eval':>5s}")
    for lib, s in report["per_library"].items():
        print(f"{lib:13s} {s['pages']:7d} {s['chars']:12,d} {s['p50']:7d} {s['p90']:7d} "
              f"{s['parts']:6d} {s['eval']:5d}")
    rng = __import__("random").Random(0)
    for r in df.sample(n=min(2, len(df)), random_state=rng.randint(0, 10**6)).itertuples():
        print(f"\n===== example {r.id} | {r.title} | {r.url} | {r.n_chars} chars =====")
        print(r.text[:1500] + ("\n[...]" if r.n_chars > 1500 else ""))


if __name__ == "__main__":
    main()
