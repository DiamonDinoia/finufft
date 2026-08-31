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

import re
import sys
import tempfile
from pathlib import Path

BLOCK = re.compile(r"^\.\.\s+literalinclude::\s*(\S+)\s*$")
OPTION = re.compile(r"^\s+:(start-after|end-before):\s*(\S+)\s*$")


def check(docs: Path) -> list[str]:
    """Return one message per unresolved include, empty when the tree is clean."""
    problems = []
    for rst in sorted(docs.rglob("*.rst")):
        lines = rst.read_text().splitlines()
        i = 0
        while i < len(lines):
            block = BLOCK.match(lines[i])
            if not block:
                i += 1
                continue
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
    return problems


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
        if check(docs):
            print("selftest: a clean tree was reported dirty", file=sys.stderr)
            return 1
        (docs / "recipe.txt").write_text("# @good_start\nkeep me\n")
        problems = check(docs)
        if len(problems) != 1 or "@good_end" not in problems[0]:
            print(
                f"selftest: a removed tag was not caught: {problems}", file=sys.stderr
            )
            return 1
        (docs / "recipe.txt").unlink()
        if not check(docs):
            print("selftest: a missing file was not caught", file=sys.stderr)
            return 1
    print("selftest: the check fails on a removed tag and on a missing file")
    return 0


def main() -> int:
    if "--selftest" in sys.argv[1:]:
        return selftest()
    problems = check(Path(__file__).resolve().parents[2] / "docs")
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} unresolved literalinclude(s)", file=sys.stderr)
        return 1
    print("every literalinclude resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
