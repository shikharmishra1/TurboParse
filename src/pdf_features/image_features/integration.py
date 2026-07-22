"""Integration: combine PdfFeatures token extraction with OpenCV figure detection.

Usage:
    from pdf_image_features import pdf_features_with_images

    pdf_features, overlaps = pdf_features_with_images("document.pdf")

    # overlaps: dict[(page_number, left, top) -> bool]
    # True if the token's bbox overlaps any detected non-text region.

    trainer = TokenTypeTrainer([pdf_features], ModelConfiguration())
    trainer.predict("model/path.model")
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfPage import PdfPage
from pdf_features.Rectangle import Rectangle

from .figure_detector import FigureDetector, PageRegions, DetectedRegion
from .page_renderer import render_page_to_image, render_page_to_gray
from .region_classifier import classify_regions


import logging
import time

logger = logging.getLogger(__name__)

# Detection DPI: render at lower resolution for dark-blob detection.
# The detector's _set_scale() adjusts thresholds proportionally, so
# results are equivalent to full-DPI detection but ~2× faster.
_DETECTION_DPI_FACTOR: float = 0.67  # 100 DPI when user DPI is 150


def pdf_features_with_images(
    pdf_path: str | Path,
    dpi: int = 150,
    use_token_mask: bool = True,
    pdf_document: object | None = None,
) -> tuple[PdfFeatures, dict[tuple[int, int, int], bool], list[DetectedRegion]]:
    """Extract PdfFeatures, detect image/figure regions, and compute token overlaps.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Render resolution (detection uses 0.67× this internally).
        use_token_mask: Use token-aware dark blob detection.
        pdf_document: Pre-opened pypdfium2.PdfDocument to reuse.

    Returns:
        (pdf_features, token_overlaps, all_regions)
    """

    fn_start = time.perf_counter()

    logger.debug("=" * 70)
    logger.debug("Starting pdf_features_with_images")
    logger.debug("PDF: %s", pdf_path)

    t0 = time.perf_counter()
    pdf_features = PdfFeatures.from_pdf_path(str(pdf_path))
    logger.debug("PdfFeatures.from_pdf_path: %.3f s", time.perf_counter() - t0)

    detector = FigureDetector()
    all_regions: list[DetectedRegion] = []

    t0 = time.perf_counter()
    doc = pdf_document if pdf_document is not None else pdfium.PdfDocument(str(pdf_path))
    doc_is_external = pdf_document is not None
    if not doc_is_external:
        logger.debug("pdfium.PdfDocument open: %.3f s", time.perf_counter() - t0)

    try:
        for page in pdf_features.pages:
            page_total = time.perf_counter()

            page_index = page.page_number - 1

            logger.debug("-" * 60)
            logger.debug(
                "Page %d (%d tokens)",
                page.page_number,
                len(page.tokens),
            )

            if page_index < 0 or page_index >= len(doc):
                logger.debug("Skipping invalid page index %d", page_index)
                continue

            t0 = time.perf_counter()
            is_scanned = len(page.tokens) == 0
            logger.debug(
                "  Scan detection: %.3f s (scanned=%s)",
                time.perf_counter() - t0,
                is_scanned,
            )

            # Strategy 1: embedded images (fast, no rendering needed)
            t0 = time.perf_counter()
            mu_regions = _detect_pdfium_images(
                doc,
                page_index,
                page.page_number,
            )
            logger.debug(
                "  PDFium image detection: %.3f s (%d regions)",
                time.perf_counter() - t0,
                len(mu_regions),
            )
            all_regions.extend(mu_regions)

            # Strategy 2: dark-blob detection — only on pages that need it.
            # Skip text-only pages with no embedded images (nothing to find).
            has_mu_images = len(mu_regions) > 0
            needs_dark_blob = has_mu_images or is_scanned

            if use_token_mask and needs_dark_blob:
                token_boxes = [token.bounding_box for token in page.tokens]
                detection_dpi = max(72, int(dpi * _DETECTION_DPI_FACTOR))

                t0 = time.perf_counter()
                # Render directly to grayscale at lower detection DPI
                page_gray = render_page_to_gray(doc, page_index, dpi=detection_dpi)
                logger.debug(
                    "  Render page (gray, %d dpi): %.3f s",
                    detection_dpi,
                    time.perf_counter() - t0,
                )

                t0 = time.perf_counter()
                page_regions = detector.detect_dark_blobs(
                    page_gray,
                    page.page_number,
                    page.page_width,
                    page.page_height,
                    token_boxes,
                    dpi=detection_dpi,
                )
                logger.debug(
                    "  Dark blob detection: %.3f s (%d regions)",
                    time.perf_counter() - t0,
                    len(page_regions.regions),
                )
            elif use_token_mask:
                # Text-only page — no dark blobs to find, create empty result
                page_regions = PageRegions(
                    page_number=page.page_number,
                    page_width=page.page_width,
                    page_height=page.page_height,
                )
                logger.debug("  Dark blob detection: skipped (text-only page)")
            elif needs_dark_blob:
                # Non-token-mask path: morphological detection
                detection_dpi = max(72, int(dpi * _DETECTION_DPI_FACTOR))
                t0 = time.perf_counter()
                page_gray = render_page_to_gray(doc, page_index, dpi=detection_dpi)
                logger.debug(
                    "  Render page (gray, %d dpi): %.3f s",
                    detection_dpi,
                    time.perf_counter() - t0,
                )
                t0 = time.perf_counter()
                page_regions = detector.detect(
                    page_gray,
                    page.page_number,
                    page.page_width,
                    page.page_height,
                    dpi=detection_dpi,
                )
                logger.debug(
                    "  Morphological detection: %.3f s (%d regions)",
                    time.perf_counter() - t0,
                    len(page_regions.regions),
                )
            else:
                # Text-only page, no token mask — nothing to detect
                page_regions = PageRegions(
                    page_number=page.page_number,
                    page_width=page.page_width,
                    page_height=page.page_height,
                )
                logger.debug("  Morphological detection: skipped (text-only page)")

            t0 = time.perf_counter()
            classify_regions(page_regions.regions)
            logger.debug(
                "  Region classification: %.3f s",
                time.perf_counter() - t0,
            )

            all_regions.extend(page_regions.regions)

            logger.debug(
                "Page %d total: %.3f s",
                page.page_number,
                time.perf_counter() - page_total,
            )

    finally:
        if not doc_is_external:
            t0 = time.perf_counter()
            doc.close()
            logger.debug("pdfium close: %.3f s", time.perf_counter() - t0)

    t0 = time.perf_counter()
    token_overlaps = _build_overlap_dict(pdf_features, all_regions)
    logger.debug(
        "_build_overlap_dict: %.3f s",
        time.perf_counter() - t0,
    )

    logger.debug(
        "Total pdf_features_with_images: %.3f s",
        time.perf_counter() - fn_start,
    )
    logger.debug("=" * 70)

    return pdf_features, token_overlaps, all_regions

def detect_figures_only(
    pdf_path: str | Path,
    dpi: int = 150,
) -> list[DetectedRegion]:
    """Run figure detection without token classification.

    Combines pypdfium2 embedded image detection (for born-digital PDFs
    with distinct image objects) and OpenCV morphological detection
    (for born-digital PDFs with thin text).

    For scanned pages or pages with full-page background images, use
    pdf_features_with_images() instead — it has access to token boxes.
    """
    detector = FigureDetector()
    all_regions: list[DetectedRegion] = []
    doc = pdfium.PdfDocument(str(pdf_path))

    try:
        for page_index in range(len(doc)):
            page_image = render_page_to_gray(doc, page_index, dpi=dpi)
            page = doc[page_index]
            page_w, page_h = page.get_size()

            # PDFium embedded image detection (always run)
            mu_regions = _detect_pdfium_images(doc, page_index, page_index)
            classify_regions(mu_regions)
            all_regions.extend(mu_regions)

            # OpenCV morphological detection (born-digital text pages)
            page_regions = detector.detect(
                page_image,
                page_index,
                int(page_w),
                int(page_h),
                dpi=dpi,
            )
            classify_regions(page_regions.regions)
            all_regions.extend(page_regions.regions)

    finally:
        doc.close()

    return all_regions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_scanned_page(doc: pdfium.PdfDocument, page_index: int) -> bool:
    """A page is 'scanned' only if it has NO text objects — just an image.

    If the page has real text blocks (born-digital or OCR overlay),
    it is NOT considered scanned, even if there is a full-page image.
    """
    page = doc[page_index]
    return page.get_textpage().count_chars() == 0


def _detect_pdfium_images(
    doc: pdfium.PdfDocument, page_index: int, page_number: int
) -> list[DetectedRegion]:
    """Extract embedded image bounding boxes via pypdfium2.

    - Scanned pages (no text, only image): report full page as ScannedPage.
    - Born-digital: report individual embedded images (skip full-page
      backgrounds when text is present).
    """
    page = doc[page_index]
    page_w, page_h = page.get_size()
    page_area = page_w * page_h

    # Check if there are text objects on the page
    has_text = page.get_textpage().count_chars() > 0

    regions: list[DetectedRegion] = []

    for obj in page.get_objects(
        filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=2
    ):
        # Get the image's transformation matrix: (a, b, c, d, e, f)
        # Unit square (0,0)-(1,1) transforms to a parallelogram
        # in PDF coordinates (bottom-left origin, Y ↑).
        matrix = obj.get_matrix()
        a, b, c, d, e, f = matrix.get()

        # Compute bounding box, converting from PDF coords (Y ↑)
        # to screen coords (top-left origin, Y ↓).
        corners = [
            (e, page_h - f),
            (a + e, page_h - (b + f)),
            (c + e, page_h - (d + f)),
            (a + c + e, page_h - (b + d + f)),
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)

        img_w = abs(x1 - x0)
        img_h = abs(y1 - y0)

        # Full-page image with NO text → truly scanned page
        if img_w * img_h > 0.6 * page_area and not has_text:
            regions.append(
                DetectedRegion(
                    bbox=Rectangle.from_width_height(
                        left=0, top=0,
                        width=int(page_w),
                        height=int(page_h),
                    ),
                    page_number=page_number,
                    area=int(page_area),
                    width=int(page_w),
                    height=int(page_h),
                    aspect_ratio=page_w / page_h
                    if page_h > 0 else 0,
                    fill_density=1.0,
                    edge_density=0.0,
                    color_variance=0.0,
                    horizontal_line_count=0,
                    vertical_line_count=0,
                    region_type="ScannedPage",
                )
            )
            continue

        # Full-page image WITH text → background, skip it
        if img_w * img_h > 0.6 * page_area and has_text:
            continue

        # Smaller embedded images
        regions.append(
            DetectedRegion(
                bbox=Rectangle.from_width_height(
                    left=int(x0), top=int(y0),
                    width=int(img_w), height=int(img_h),
                ),
                page_number=page_number,
                area=int(img_w * img_h),
                width=int(img_w),
                height=int(img_h),
                aspect_ratio=img_w / img_h if img_h > 0 else 0,
                fill_density=1.0,
                edge_density=0.0,
                color_variance=0.0,
                horizontal_line_count=0,
                vertical_line_count=0,
                region_type="Figure",
            )
        )

    return regions


def _build_overlap_dict(
    pdf_features: PdfFeatures,
    regions: list[DetectedRegion],
) -> dict[tuple[int, int, int], bool]:
    """Create a dict: (page_number, left, top) → overlaps_any_region.

    ScannedPage regions are excluded — they cover the entire page and
    would mark every token as overlapping, which isn't informative.
    """
    # Filter out ScannedPage regions for overlap calculation
    non_scanned = [r for r in regions if r.region_type != "ScannedPage"]

    regions_by_page: dict[int, list[DetectedRegion]] = {}
    for region in non_scanned:
        regions_by_page.setdefault(region.page_number, []).append(region)

    overlaps: dict[tuple[int, int, int], bool] = {}

    for page in pdf_features.pages:
        page_regions = regions_by_page.get(page.page_number, [])
        for token in page.tokens:
            key = (
                token.page_number,
                int(token.bounding_box.left),
                int(token.bounding_box.top),
            )
            overlaps[key] = _token_overlaps_any_region(token.bounding_box, page_regions)

    return overlaps


def _token_overlaps_any_region(
    token_box: Rectangle,
    regions: list[DetectedRegion],
) -> bool:
    """Check if token_box intersects any detected region."""
    for region in regions:
        if _rectangles_intersect(token_box, region.bbox):
            return True
    return False


def _rectangles_intersect(a: Rectangle, b: Rectangle) -> bool:
    """True if two rectangles intersect (including touching edges)."""
    return not (
        a.right < b.left
        or a.left > b.right
        or a.bottom < b.top
        or a.top > b.bottom
    )
