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


def compile_ui(ui_file: Path, out_file: Path) -> None:
    buf = io.StringIO()
    buf.write(HEADER.format(ui_name=ui_file.name))
    uic.compileUi(str(ui_file), buf)
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
