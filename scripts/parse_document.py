from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.PDF_parse.parser import parse_pdf_document, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse a PDF into section-aware JSON chunks.")
    parser.add_argument("input_pdf", help="Path to the input PDF")
    parser.add_argument("--output", required=True, help="Path to write the output JSON")
    parser.add_argument(
        "--heading-diagnostics",
        action="store_true",
        help="Include heading candidate scores for tuning.",
    )
    args = parser.parse_args()

    output = parse_pdf_document(args.input_pdf, include_heading_diagnostics=args.heading_diagnostics)
    write_json(output, args.output)
    return 0 if output["document"]["parse_status"] in {"OK", "OCR_REQUIRED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
