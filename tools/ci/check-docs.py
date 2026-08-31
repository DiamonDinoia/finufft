#!/usr/bin/env python3
"""Every literalinclude in docs/ has to resolve: the file exists, and the tags the
block names are present in it.

Sphinx answers a missing tag with a warning and renders the wrong span, and
nothing fails on a warning: docs/Makefile passes no -W, and .readthedocs.yaml
does not set sphinx.fail_on_warning. So a renamed or deleted
"# sphinx tag (don't remove): @name" would quietly publish the whole file, or
half of it, in place of the recipe.

The recipes need no version check: they fetch master, which is exactly what
every CI cell builds and runs.

    python3 tools/ci/check-docs.py             # check the tree
    python3 tools/ci/check-docs.py --selftest  # prove the check can fail
"""

import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

BLOCK = re.compile(r"^\.\.\s+literalinclude::\s*(\S+)\s*$")
OPTION = re.compile(r"^\s+:(start-after|end-before):\s*(\S+)\s*$")


def check(docs: Path) -> tuple[list[str], int]:
    """One message per unresolved include, plus how many directives were seen."""
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
            # The options of a directive are the indented lines under it.
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


def selftest() -> int:
    """The control: the check has to fail on a tree whose tag was removed."""
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
        # The two guards in run(): an absent docs root, and a root whose .rst
        # files hold no directives. Each guard owns a message, and the message
        # is the assertion: without the first, a missing root reports "no
        # directives" instead of "does not exist", and still exits nonzero.
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
    print(
        "selftest: the check fails on a removed tag, a missing file,"
        " a missing docs root and a directive-less root"
    )
    return 0


def run(docs: Path) -> int:
    """The guards main() relies on, around check(), so selftest can reach them."""
    # An empty result set is a bug, not a clean tree: rglob on a missing or moved
    # docs root yields nothing instead of failing, which used to print success.
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
    return run(Path(__file__).resolve().parents[2] / "docs")


if __name__ == "__main__":
    sys.exit(main())
