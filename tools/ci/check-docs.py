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
OPTION = re.compile(r"^\s+:(start-after|end-before):\s*(.+?)\s*$")
# The tag convention: a docs page embeds a region of a source file marked
#     <comment> sphinx tag (don't remove): @<name>_start ... @<name>_end
# <name> is lower snake_case, names its own file or the thing embedded, and is
# unique tree-wide. tags() enforces the pairing, the uniqueness and the prose
# prefix, so a region cannot be marked one way here and another way there.
MARKER = re.compile(r"sphinx tag[^@]*@(\w+)_(start|end)\b")
TAG = re.compile(r"^@(\w+)_(start|end)$")
# Suffixes a marked region can live in. A tag outside these is invisible to the gate.
TAGGED = (
    "*.h",
    "*.hpp",
    "*.c",
    "*.cpp",
    "*.cu",
    "*.cuh",
    "*.f",
    "*.f90",
    "*.m",
    "*.py",
    "*.sh",
    "*.yml",
    "*.txt",
    "*.cmake",
    "makefile",
    "Jenkinsfile",
    "make.inc*",
)
# Directories with no sources of their own: build trees, vendored copies, the docs,
# and .claude, which holds nested git worktrees whose tags would look like duplicates.
SKIP = (".git", ".claude", "docs", "__pycache__", "_deps", "_html")
COPIES = ("examples/quick-start/*/main.cpp", "examples/quick-start/cuda/*/main.cpp")
# Pages the docs embed verbatim from a command's output, so the command stays the one
# place the text is written.
GENERATED = {"docs/makefile.doc": ("make", "usage")}
# Directories whose files are programs or scripts, so a docs embed of one is a claim
# that CI builds or runs it.
RUNNABLE = ("examples/", "tutorial/", "tools/ci/", "test/")
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
        # Otherwise the build file beside it must name it, glob its suffix, or name the
        # stem, which is how a target list spells its sources, one per line or several
        # to a line. The stem match needs a boundary: simple1d1c must not match the
        # simple1d1cf that a longer target name spells.
        build = root / parent / "CMakeLists.txt"
        if build.is_file():
            text = build.read_text()
            stem, suffix = Path(rel).stem, Path(rel).suffix
            bounded = rf"(?<![\w.-]){re.escape(stem)}(?![\w.-])"
            if name in text or f"*{suffix}" in text or re.search(bounded, text):
                continue
        problems.append(f"{where}: {rel} is embedded but not executed by CI")
    return problems, exempt


def tags(root: Path) -> tuple[list[str], int]:
    """Enforce the tag convention documented beside MARKER."""
    marked: dict[str, list[tuple[str, str, int]]] = {}
    plain: dict[str, str] = {}
    for pattern in TAGGED:
        for f in sorted(root.rglob(pattern)):
            if not f.is_file() or set(f.relative_to(root).parts) & set(SKIP):
                continue
            rel = f.relative_to(root).as_posix()
            for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                found = MARKER.search(line)
                if found:
                    marked.setdefault(found.group(1), []).append(
                        (rel, found.group(2), n)
                    )
                elif "@" in line:
                    for name in re.findall(r"@(\w+)_(?:start|end)\b", line):
                        plain.setdefault(name, f"{rel}:{n}")

    problems = []
    for name, sites in sorted(marked.items()):
        files = {rel for rel, _, _ in sites}
        if len(files) > 1:
            problems.append(
                f"@{name}_start is marked in {len(files)} files: {sorted(files)}"
            )
            continue
        rel = files.pop()
        kinds = sorted(kind for _, kind, _ in sites)
        if kinds != ["end", "start"]:
            problems.append(
                f"{rel}: @{name}_* is marked {kinds}, not one start and one end"
            )
            continue
        lines = {kind: n for _, kind, n in sites}
        if lines["start"] > lines["end"]:
            problems.append(
                f"{rel}:{lines['start']}: @{name}_end comes before @{name}_start"
            )

    used = set()
    for rst in sorted((root / "docs").rglob("*.rst")):
        for line in rst.read_text().splitlines():
            option = OPTION.match(line)
            if not option:
                continue
            tag = TAG.match(option.group(2))
            if not tag:
                problems.append(f"{rst}: {option.group(2)} is not an @<name>_start tag")
                continue
            name = tag.group(1)
            used.add(name)
            if name in marked:
                continue
            where = plain.get(name, "nowhere the gate can see")
            problems.append(
                f"{rst}: @{name}_{tag.group(2)} at {where} lacks the"
                " `sphinx tag (don't remove):` prefix"
            )
    for name in sorted(set(marked) - used):
        rel = marked[name][0][0]
        problems.append(f"{rel}: @{name}_start marks a region no docs page embeds")
    return problems, len(used)


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
        # A target list names stems, several to a line. The stem match must respect a
        # boundary, or a prefix of a longer target name passes for free.
        (tree / "examples/CMakeLists.txt").write_text(
            "set(EXAMPLES_C run_me orphanx)\n"
        )
        problems, _ = executed(tree)
        if len(problems) != 1 or "not executed by CI" not in problems[0]:
            print(
                f"selftest: a stem that is only a prefix passed: {problems}",
                file=sys.stderr,
            )
            return 1
        (tree / "examples/CMakeLists.txt").write_text("set(EXAMPLES_C run_me orphan)\n")
        if executed(tree)[0]:
            print(
                "selftest: a stem in a space-separated target list was not found",
                file=sys.stderr,
            )
            return 1
        (tree / "examples/CMakeLists.txt").unlink()
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
        tagged = Path(tmp) / "tagged"
        (tagged / "docs").mkdir(parents=True)
        (tagged / "src").mkdir()
        marks = "// sphinx tag (don't remove): @one_{}\n"
        good = marks.format("start") + "code\n" + marks.format("end")
        (tagged / "src/a.cpp").write_text(good)
        (tagged / "docs/page.rst").write_text(
            ".. literalinclude:: ../src/a.cpp\n"
            "   :start-after: @one_start\n"
            "   :end-before: @one_end\n"
        )
        problems, used = tags(tagged)
        if problems or used != 1:
            print(
                f"selftest: a conforming tag was reported: {problems}", file=sys.stderr
            )
            return 1
        (tagged / "src/a.cpp").write_text(marks.format("start") + "code\n")
        if not any("not one start and one end" in p for p in tags(tagged)[0]):
            print("selftest: an unpaired tag was not caught", file=sys.stderr)
            return 1
        (tagged / "src/a.cpp").write_text(good)
        (tagged / "src/b.cpp").write_text(good)
        if not any("is marked in 2 files" in p for p in tags(tagged)[0]):
            print("selftest: a duplicated tag name was not caught", file=sys.stderr)
            return 1
        (tagged / "src/b.cpp").unlink()
        (tagged / "src/a.cpp").write_text("// @one_start\ncode\n// @one_end\n")
        if not any("lacks the" in p for p in tags(tagged)[0]):
            print("selftest: an unmarked tag was not caught", file=sys.stderr)
            return 1
        (tagged / "src/a.cpp").write_text(good)
        nested = tagged / ".claude/worktrees/agent/src"
        nested.mkdir(parents=True)
        (nested / "a.cpp").write_text(good)
        problems, used = tags(tagged)
        if problems or used != 1:
            print(
                f"selftest: a nested worktree was not skipped: {problems}",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(tagged / ".claude")
        (tagged / "docs/page.rst").write_text("prose only\n")
        if not any("no docs page embeds" in p for p in tags(tagged)[0]):
            print("selftest: an unembedded region was not caught", file=sys.stderr)
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
        " and an embedded file nothing in CI runs, a hand-edited c*.doc page,"
        " an unpaired, a duplicated, an unmarked and an unembedded sphinx tag,"
        " and it ignores a tag inside a nested git worktree,"
        " and it does not accept a target stem that is only a prefix"
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
    untagged, used = tags(root)
    problems = (
        copies(root) + generated(root) + scripted(root / "docs") + unrun + untagged
    )
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print("every recipe copy is identical")
    print("every generated page matches the command that writes it")
    print("every c*.doc page matches docs/makecdocs.sh")
    print(f"{used} sphinx tags pair up, are unique and are embedded once")
    for line in exempt:
        print(line)
    print("every other runnable embed is named by a CI configuration")
    return status


if __name__ == "__main__":
    sys.exit(main())
