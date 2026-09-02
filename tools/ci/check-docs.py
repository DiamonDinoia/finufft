#!/usr/bin/env python3
"""Check the docs literalincludes and the recipe copies: usage: check-docs.py [--selftest]."""

import contextlib
import hashlib
import io
import shutil
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
# Directories whose files are programs or scripts, so a docs embed of one is a claim
# that CI builds or runs it.
RUNNABLE = ("examples/", "tutorial/", "tools/ci/")
# Files that name what CI does, so a runnable embed must appear in one of them.
RUNNERS = (
    ".github/workflows/*.yml",
    "Jenkinsfile",
    "makefile",
    "tools/ci/*.sh",
    "CMakeLists.txt",
    "*/CMakeLists.txt",
    "*/*/CMakeLists.txt",
    "*/*/*/CMakeLists.txt",
)
# Runnable embeds that nothing builds or runs. The gate prints them, so a silent
# exemption cannot grow.
NOT_RUN = {
    "tutorial/nfft2d1_test.c": "needs libnfft3-dev, which no CI image installs",
    "tutorial/migrate2d1_test.c": "needs libnfft3-dev, which no CI image installs",
    "tutorial/samplegrf1d.m": "the Octave arms run matlab/test, not the tutorials",
}


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


def targets(docs: Path, root: Path) -> list[tuple[str, str]]:
    """Every literalinclude target, as (where, path relative to root)."""
    found = []
    for rst in sorted(docs.rglob("*.rst")):
        for i, line in enumerate(rst.read_text().splitlines()):
            block = BLOCK.match(line)
            if not block:
                continue
            target = (rst.parent / block.group(1)).resolve()
            with contextlib.suppress(ValueError):
                found.append((f"{rst}:{i + 1}", target.relative_to(root).as_posix()))
    return found


def executed(root: Path) -> tuple[list[str], list[str]]:
    """Check that every runnable embed is named by something CI runs."""
    docs = root / "docs"
    texts = [
        f.read_text(errors="replace")
        for pattern in RUNNERS
        for f in sorted(root.glob(pattern))
        if f.is_file()
    ]
    problems, exempt = [], []
    for where, rel in sorted(set(targets(docs, root))):
        if not rel.startswith(RUNNABLE):
            continue
        if rel in NOT_RUN:
            exempt.append(f"NOT RUN {rel}: {NOT_RUN[rel]}")
            continue
        name = Path(rel).name
        parent = rel.rsplit("/", 1)[0]
        if any(rel in text for text in texts):
            continue
        # A build file is reached through its directory: that is what a CI script or a
        # parent project names, never the file itself.
        if name in ("CMakeLists.txt", "Makefile", "makefile"):
            if any(parent in text for text in texts):
                continue
        # Otherwise the build file beside it must name it, or glob its suffix.
        build = root / parent / "CMakeLists.txt"
        if build.is_file():
            text = build.read_text()
            if name in text or f"*{Path(rel).suffix}" in text:
                continue
        problems.append(f"{where}: {rel} is embedded but not executed by CI")
    return problems, exempt


def scripted(docs: Path) -> list[str]:
    """docs/makecdocs.sh writes the c*.doc pages from the *.docsrc beside them. Run it
    on a copy and diff, so a hand edit to a generated page fails the gate."""
    script = docs / "makecdocs.sh"
    pages = sorted(docs.glob("c*.doc"))
    if not script.is_file():
        return [f"{script} is missing"]
    if not pages:
        return [f"{docs}/c*.doc: no generated pages to check"]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for source in [script, *docs.glob("*.docsrc")]:
            shutil.copy2(source, work / source.name)
        result = subprocess.run(
            ["bash", script.name], cwd=work, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return [f"{script}: failed: {result.stderr.strip()[:200]}"]
        problems = []
        for page in pages:
            fresh = work / page.name
            if not fresh.is_file():
                problems.append(f"{page}: {script.name} no longer writes it")
            elif fresh.read_text() != page.read_text():
                problems.append(
                    f"{page}: drifted from {script.name}, regenerate with `make docs`"
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
        tree = Path(tmp) / "exec"
        (tree / "docs").mkdir(parents=True)
        (tree / "examples").mkdir()
        (tree / "examples/run_me.cpp").write_text("int main() { return 0; }\n")
        (tree / "examples/orphan.cpp").write_text("int main() { return 0; }\n")
        (tree / "CMakeLists.txt").write_text("add_executable(a examples/run_me.cpp)\n")
        (tree / "docs/page.rst").write_text(
            ".. literalinclude:: ../examples/run_me.cpp\n"
        )
        problems, exempt = executed(tree)
        if problems or exempt:
            print(
                f"selftest: an executed embed was reported: {problems}", file=sys.stderr
            )
            return 1
        (tree / "docs/page.rst").write_text(
            ".. literalinclude:: ../examples/orphan.cpp\n"
        )
        problems, _ = executed(tree)
        if len(problems) != 1 or "not executed by CI" not in problems[0]:
            print(
                f"selftest: an unexecuted embed was not caught: {problems}",
                file=sys.stderr,
            )
            return 1
        NOT_RUN["examples/orphan.cpp"] = "selftest"
        try:
            problems, exempt = executed(tree)
        finally:
            del NOT_RUN["examples/orphan.cpp"]
        if problems or len(exempt) != 1:
            print(
                f"selftest: the allow-list did not exempt: {problems} {exempt}",
                file=sys.stderr,
            )
            return 1
        script = Path(tmp) / "cdocs"
        script.mkdir()
        (script / "makecdocs.sh").write_text(
            'for i in *.docsrc; do sed s/@F/finufft/ "$i" > "${i/.docsrc/.doc}"; done\n'
        )
        (script / "c1d.docsrc").write_text("@F1d1\n")
        (script / "c1d.doc").write_text("finufft1d1\n")
        if scripted(script):
            print(
                "selftest: a current scripted page was reported dirty", file=sys.stderr
            )
            return 1
        (script / "c1d.doc").write_text("hand edited\n")
        if not any("drifted" in p for p in scripted(script)):
            print(
                "selftest: a hand-edited scripted page was not caught", file=sys.stderr
            )
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
        " and an embedded file nothing in CI runs, plus a hand-edited c*.doc page"
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
    unrun, exempt = executed(root)
    problems = copies(root) + generated(root) + scripted(root / "docs") + unrun
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print("every recipe copy is identical")
    print("every generated page matches the command that writes it")
    print("every c*.doc page matches docs/makecdocs.sh")
    for line in exempt:
        print(line)
    print("every other runnable embed is named by a CI configuration")
    return status


if __name__ == "__main__":
    sys.exit(main())
