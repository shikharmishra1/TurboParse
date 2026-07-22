"""
PDF token-type prediction with layout analysis (regions + TSR tables).

Usage:
    from predict import predict_with_layout, predict_tokens_only

    pdf_features, predictions, all_regions, table_tsr = predict_with_layout("doc.pdf")
    # Or lightweight: just token types, no image/table detection
    predictions = predict_tokens_only("doc.pdf")
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pypdfium2 as pdfium
import xml.etree.ElementTree as ET

from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfPage import PdfPage
from Token import Token
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from pdf_features.image_features import pdf_features_with_images
from pdf_features.image_features.figure_detector import DetectedRegion
from pdf_features.image_features.page_renderer import render_page_to_image
from tsr import TSR


def predict_tokens_only(
    pdf_path: str | Path,
    model_path: str | None = None,
) -> list[dict]:
    """Predict token types only — no image region detection, no TSR.

    Returns list of dicts (Token.to_dict() format).
    """
    pdf_features = PdfFeatures.from_pdf_path(str(pdf_path))
    trainer = TokenTypeTrainer([pdf_features], ModelConfiguration())
    trainer.predict(model_path)

    predictions: list[dict] = []
    for token in trainer.loop_tokens():
        token_type = TokenType.from_index(token.prediction)
        predictions.append(Token.from_pdf_token(token, token_type).to_dict())

    return predictions


def predict_with_layout(
    pdf_path: str | Path,
    pdf_features: PdfFeatures | None = None,
    trainer: TokenTypeTrainer | None = None,
    token_image_overlaps: dict | None = None,
    all_regions: list[DetectedRegion] | None = None,
    model_path: str | None = None,
    dpi: int = 150,
    pdf_document: object | None = None,
) -> tuple[PdfFeatures, list[dict], list[DetectedRegion], list[dict]]:
    """Full pipeline: token types + image region detection + TSR table structure.

    Args:
        pdf_path: Path to a PDF file.
        model_path: Path to trained .model file (None = default).
        dpi: Render resolution for image processing.
        all_regions: Pre-computed regions from pdf_features_with_images().
            When provided, skips re-detection — only renders pages with
            Table regions for TSR cropping.
        pdf_document: Pre-opened pypdfium2.PdfDocument. When provided,
            reused for TSR rendering instead of opening a new one.

    Returns:
        (pdf_features, predictions, all_regions, table_tsr_results)
    """
    pdf_path = str(pdf_path)

    # ── Step 1: extract tokens + image overlaps + regions ─────────
    if pdf_features is None or token_image_overlaps is None or all_regions is None:
        pdf_features, token_image_overlaps, all_regions = pdf_features_with_images(pdf_path, dpi=dpi)

    # ── Step 2: predict token types ───────────────────────────────
    if trainer is None:
        trainer = TokenTypeTrainer(
            [pdf_features], ModelConfiguration(), token_image_overlaps=token_image_overlaps,
        )

    trainer.predict(model_path)

    predictions: list[dict] = []
    for token in trainer.loop_tokens():
        token_type = TokenType.from_index(token.prediction)
        predictions.append(Token.from_pdf_token(token, token_type).to_dict())

    # ── Step 3: TSR on Table regions (only render pages that have tables) ──
    tsr = TSR()
    table_tsr_results: list[dict] = []
    scale = dpi / 72.0

    # Find which pages have Table regions
    table_pages: set[int] = set()
    for r in all_regions:
        if r.region_type == "Table":
            table_pages.add(r.page_number)

    if table_pages:
        doc = pdf_document if pdf_document is not None else pdfium.PdfDocument(pdf_path)
        doc_is_external = pdf_document is not None
        try:
            for page in pdf_features.pages:
                if page.page_number not in table_pages:
                    continue

                pi = page.page_number - 1
                if pi < 0 or pi >= len(doc):
                    continue

                page_img = render_page_to_image(doc, pi, dpi=dpi)
                page_tables = [r for r in all_regions
                               if r.page_number == page.page_number and r.region_type == "Table"]

                for r in page_tables:
                    x1 = int(r.bbox.left * scale)
                    y1 = int(r.bbox.top * scale)
                    x2 = int(r.bbox.right * scale)
                    y2 = int(r.bbox.bottom * scale)

                    h_img, w_img = page_img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    table_crop = page_img[y1:y2, x1:x2]
                    table_bgr = cv2.cvtColor(table_crop, cv2.COLOR_RGB2BGR)

                    try:
                        xml_str, _ = tsr.predict(table_bgr)
                        root = ET.fromstring(xml_str)
                        cells_pdf = []
                        for cell in root.findall("cell"):
                            bb = cell.find("boundingbox")
                            if bb is not None:
                                cells_pdf.append({
                                    "row": int(cell.get("row", 0)),
                                    "col": int(cell.get("column", 0)),
                                    "x": float(bb.get("x", 0)) / scale,
                                    "y": float(bb.get("y", 0)) / scale,
                                    "w": float(bb.get("w", 0)) / scale,
                                    "h": float(bb.get("h", 0)) / scale,
                                })
                        table_tsr_results.append({
                            "page": page.page_number,
                            "bbox": {
                                "left": r.bbox.left, "top": r.bbox.top,
                                "right": r.bbox.right, "bottom": r.bbox.bottom,
                            },
                            "width": r.width, "height": r.height,
                            "num_cells": len(cells_pdf),
                            "cells": cells_pdf,
                            "dpi": dpi,
                            "xml": xml_str,
                        })
                    except Exception:
                        pass
        finally:
            if not doc_is_external:
                doc.close()

    # ── Step 4: deduplicate contained regions ─────────────────────
    all_regions = _deduplicate_contained_regions(all_regions)

    return pdf_features, predictions, all_regions, table_tsr_results


# ── region deduplication ──────────────────────────────────────────────

# Types that are "image-like" — if one fully contains another of these,
# keep only the parent.
_IMAGE_TYPES = {"Photo", "Figure", "Picture", "Chart", "Logo"}


def _region_contains(outer, inner) -> bool:
    """True if outer's bbox fully contains inner's bbox."""
    return (
        outer.bbox.left <= inner.bbox.left
        and outer.bbox.top <= inner.bbox.top
        and outer.bbox.right >= inner.bbox.right
        and outer.bbox.bottom >= inner.bbox.bottom
    )


def _deduplicate_contained_regions(
    regions: list[DetectedRegion],
) -> list[DetectedRegion]:
    """Remove image regions that are fully contained within a larger
    image region of the same/compatible type.  Tables are never removed."""
    if not regions:
        return regions

    # Group by page
    by_page: dict[int, list[DetectedRegion]] = {}
    for r in regions:
        by_page.setdefault(r.page_number, []).append(r)

    kept: list[DetectedRegion] = []
    for page_regions in by_page.values():
        # Separate image-like from everything else
        images = [r for r in page_regions if r.region_type in _IMAGE_TYPES]
        others = [r for r in page_regions if r.region_type not in _IMAGE_TYPES]
        kept.extend(others)  # tables, formulas, headers etc. always kept

        # Sort images largest-first so parents come before children
        images.sort(key=lambda r: r.area, reverse=True)

        survivors: list[DetectedRegion] = []
        for r in images:
            # Keep r unless it's fully contained in an already-kept image
            if any(_region_contains(k, r) for k in survivors):
                continue
            survivors.append(r)
        kept.extend(survivors)

    return kept


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) >= 3 else None

    pdf_features, predictions, regions, tsr_results = predict_with_layout(
        pdf_path, model_path,
    )

    print(f"Pages: {len(pdf_features.pages)}")
    print(f"Predictions: {len(predictions)}")
    print(f"Regions: {len(regions)}")
    print(f"TSR tables: {len(tsr_results)}")

    # Also run markdown conversion
    from converters.markdown_converter import predictions_to_markdown

    md = predictions_to_markdown(pdf_features, predictions, regions, tsr_results, pdf_path=pdf_path)
    out = Path(pdf_path).with_suffix(".md")
    out.write_text(md, encoding="utf-8")
    print(f"Markdown saved to {out}")
