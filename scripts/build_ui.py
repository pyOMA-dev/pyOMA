#!/usr/bin/env python
"""
Compile pyOMA/GUI/ui/*.ui files into pyOMA/GUI/generated/ui_*.py modules.

Run this after editing a .ui file in Qt Designer::

    python scripts/build_ui.py

Use --check in CI / pre-commit to verify the committed generated/ files are
still in sync with their .ui sources, without writing anything::

    python scripts/build_ui.py --check
"""
import argparse
import filecmp
import io
import sys
import tempfile
from pathlib import Path

from PyQt6 import uic

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = REPO_ROOT / 'pyOMA' / 'GUI' / 'ui'
GENERATED_DIR = REPO_ROOT / 'pyOMA' / 'GUI' / 'generated'

HEADER = (
    "# DO NOT EDIT — generated from {ui_name} by scripts/build_ui.py\n"
)


class _RenamedFile:
    """Wraps an open file object, overriding .name.

    uic.compileUi() embeds ``uifile.name`` (or the path string itself) in a
    "Form implementation generated from reading ui file '...'" comment. If we
    handed it ui_file's absolute path, that comment would differ depending on
    where the repo happens to be checked out, making --check always report
    every file as stale on any machine other than the one that last committed
    it. Giving it a stable repo-relative name instead keeps the generated
    output identical everywhere.
    """

    def __init__(self, fileobj, name):
        self._fileobj = fileobj
        self.name = name

    def __getattr__(self, attr):
        return getattr(self._fileobj, attr)


def compile_ui(ui_file: Path, out_file: Path) -> None:
    buf = io.StringIO()
    buf.write(HEADER.format(ui_name=ui_file.name))
    with open(ui_file, encoding='utf-8') as f:
        renamed = _RenamedFile(f, ui_file.relative_to(REPO_ROOT).as_posix())
        uic.compileUi(renamed, buf)
    out_file.write_text(buf.getvalue())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check', action='store_true',
        help="don't write generated/ files; exit non-zero if they'd change")
    args = parser.parse_args()

    ui_files = sorted(UI_DIR.glob('*.ui'))
    if not ui_files:
        print(f"No .ui files found in {UI_DIR}", file=sys.stderr)
        return 1

    if args.check:
        with tempfile.TemporaryDirectory() as tmpdir:
            mismatches = []
            for ui_file in ui_files:
                generated_name = f'ui_{ui_file.stem}.py'
                committed = GENERATED_DIR / generated_name
                candidate = Path(tmpdir) / generated_name
                compile_ui(ui_file, candidate)
                if not committed.exists() or not filecmp.cmp(
                        committed, candidate, shallow=False):
                    mismatches.append(generated_name)
            if mismatches:
                print(
                    "Generated UI files are out of sync with their .ui "
                    f"sources: {', '.join(mismatches)}\n"
                    "Run `python scripts/build_ui.py` and commit the result.",
                    file=sys.stderr)
                return 1
        print("All generated UI files are up to date.")
        return 0

    GENERATED_DIR.mkdir(exist_ok=True)
    for ui_file in ui_files:
        generated_name = f'ui_{ui_file.stem}.py'
        out_file = GENERATED_DIR / generated_name
        compile_ui(ui_file, out_file)
        print(f"Generated {out_file.relative_to(REPO_ROOT)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
