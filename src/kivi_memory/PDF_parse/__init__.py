"""Standalone PDF parsing and section-aware chunking.

This package is intentionally isolated from semantic memory, embeddings,
retrieval, writeback, and database code.
"""

from .parser import parse_pdf_document

__all__ = ["parse_pdf_document"]
