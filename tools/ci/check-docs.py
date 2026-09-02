#!/usr/bin/env python3
"""Check the docs literalincludes and the recipe copies: usage: check-docs.py [--selftest]."""

import contextlib
import hashlib
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BLOCK = re.compile(r"^\.\.\s+literalinclude::\s*(\S+)\s*$")
OPTION = re.compile(r"^\s+:(start-after|end-before):\s*(\S+)\s*$")
COPIES = ("examples/quick-start/*/main.cpp", "examples/quick-start/cuda/*/main.cpp")
# Pages the docs embed verbatim from a command's output, so the command stays the one
# place the text is written.
GENERATED = {"docs/makefile.doc": ("make", "usage")}


def check(docs: Path) -> tuple[list[str], int]:
    problems = []
    count = 0
    for rst in sorted(docs.rglob("*.rst")):
        lines = rst.read_text().splitlines()
        i = 0
        while i < len(lines):
            block = BLOCK.match(lines[i])
            if not block:
                i += 1
                continue
            count += 1
            where = f"{rst}:{i + 1}"
            options, j = [], i + 1
            while j < len(lines) and lines[j].strip() and lines[j][:1].isspace():
                option = OPTION.match(lines[j])
                if option:
                    options.append(option.groups())
                j += 1
            target = (rst.parent / block.group(1)).resolve()
            if not target.is_file():
                problems.append(
                    f"{where}: literalinclude names no such file: {block.group(1)}"
                )
            else:
                text = target.read_text()
                for name, tag in options:
                    if tag not in text:
                        problems.append(
                            f"{where}: {block.group(1)} has no {name} tag {tag}"
                        )
            i = j
    return problems, count


def copies(root: Path) -> list[str]:
    problems = []
    for pattern in COPIES:
        files = sorted(root.glob(pattern))
        if len(files) < 2:
            problems.append(
                f"{pattern}: fewer than two copies to compare: {len(files)}"
            )
            continue
        digests = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        if len(set(digests.values())) > 1:
            listing = ", ".join(
                f"{f.relative_to(root)}={d[:8]}" for f, d in digests.items()
            )
            problems.append(f"{pattern}: the copies have drifted: {listing}")
    return problems


def generated(root: Path, pages: dict = GENERATED) -> list[str]:
    problems = []
    for page, command in pages.items():
        target = root / page
        printed = " ".join(command)
        if not target.is_file():
            problems.append(f"{page}: the generated page is missing")
            continue
        try:
            result = subprocess.run(
                command, cwd=root, capture_output=True, text=True, timeout=120
            )
        except (OSError, subprocess.SubprocessError) as exc:
            problems.append(f"{page}: `{printed}` did not run: {exc}")
            continue
        if result.returncode != 0:
            problems.append(
                f"{page}: `{printed}` failed: {result.stderr.strip()[:200]}"
            )
        elif result.stdout != target.read_text():
            problems.append(
                f"{page}: drifted from `{printed}`, regenerate with"
                f" `{printed} > {page}`"
            )
    return problems


def selftest() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "docs"
        docs.mkdir()
        (docs / "recipe.txt").write_text("# @good_start\nkeep me\n# @good_end\n")
        (docs / "page.rst").write_text(
            ".. literalinclude:: recipe.txt\n"
            "   :start-after: @good_start\n"
            "   :end-before: @good_end\n"
        )
        problems, _ = check(docs)
        if problems:
            print("selftest: a clean tree was reported dirty", file=sys.stderr)
            return 1
        (docs / "recipe.txt").write_text("# @good_start\nkeep me\n")
        problems, _ = check(docs)
        if len(problems) != 1 or "@good_end" not in problems[0]:
            print(
                f"selftest: a removed tag was not caught: {problems}", file=sys.stderr
            )
            return 1
        (docs / "recipe.txt").unlink()
        problems, _ = check(docs)
        if not problems:
            print("selftest: a missing file was not caught", file=sys.stderr)
            return 1
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status = run(Path(tmp) / "no-docs")
        if status == 0 or "does not exist" not in err.getvalue():
            print("selftest: a missing docs root was not caught", file=sys.stderr)
            return 1
        bare = Path(tmp) / "bare-docs"
        bare.mkdir()
        (bare / "page.rst").write_text("prose, no literals to include\n")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status = run(bare)
        if status == 0 or "no literalinclude directives" not in err.getvalue():
            print(
                "selftest: a docs root with no directives was not caught",
                file=sys.stderr,
            )
            return 1
        recipes = Path(tmp) / "tree"
        for name in ("a", "b"):
            for group in ("", "cuda/"):
                d = recipes / "examples/quick-start" / group / name
                d.mkdir(parents=True)
                (d / "main.cpp").write_text("int main() { return 0; }\n")
        if copies(recipes):
            print("selftest: identical copies were reported dirty", file=sys.stderr)
            return 1
        (recipes / "examples/quick-start/b/main.cpp").write_text("int main() {}\n")
        problems = copies(recipes)
        if len(problems) != 1 or "drifted" not in problems[0]:
            print(
                f"selftest: a drifted copy was not caught: {problems}", file=sys.stderr
            )
            return 1
        (recipes / "examples/quick-start/b/main.cpp").unlink()
        if not any("fewer than two" in p for p in copies(recipes)):
            print("selftest: a vanished copy was not caught", file=sys.stderr)
            return 1
        pages = {"page.doc": ("echo", "hi")}
        gen = Path(tmp) / "gen"
        gen.mkdir()
        (gen / "page.doc").write_text("hi\n")
        if generated(gen, pages):
            print(
                "selftest: a current generated page was reported dirty", file=sys.stderr
            )
            return 1
        (gen / "page.doc").write_text("stale\n")
        if not any("drifted" in p for p in generated(gen, pages)):
            print("selftest: a stale generated page was not caught", file=sys.stderr)
            return 1
        (gen / "page.doc").unlink()
        if not any("missing" in p for p in generated(gen, pages)):
            print("selftest: a vanished generated page was not caught", file=sys.stderr)
            return 1
    print(
        "selftest: the check fails on a removed tag, a missing file,"
        " a missing docs root, a directive-less root, a drifted recipe copy"
        " and a vanished one, plus a stale and a vanished generated page"
    )
    return 0


def run(docs: Path) -> int:
    if not docs.is_dir():
        print(f"docs directory does not exist: {docs}", file=sys.stderr)
        return 1
    problems, count = check(docs)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} unresolved literalinclude(s)", file=sys.stderr)
        return 1
    if not count:
        print(f"no literalinclude directives under {docs}", file=sys.stderr)
        return 1
    print("every literalinclude resolves")
    return 0


def main() -> int:
    if "--selftest" in sys.argv[1:]:
        return selftest()
    root = Path(__file__).resolve().parents[2]
    status = run(root / "docs")
    problems = copies(root) + generated(root)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print("every recipe copy is identical")
    print("every generated page matches the command that writes it")
    return status


if __name__ == "__main__":
    sys.exit(main())
