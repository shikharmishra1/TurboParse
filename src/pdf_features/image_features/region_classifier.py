"""Heuristic classifier for detected non-text regions.

Classifies regions into: Photo, Chart, Logo, Formula, Table, Figure.

Uses cheap-to-compute features:
  - color_variance: high → photo; low → chart/logo/formula
  - edge_density: high → chart/photo/table; low → logo
  - horizontal/vertical line count: high → table
  - aspect_ratio: wide → logo/table; tall → formula
  - fill_density: dense → photo; sparse → chart

Also supports token-type-based override: when token predictions are
available, a region whose overlapping tokens are predominantly of a
specific type (e.g., TABLE) will have its classification overridden.
"""

from __future__ import annotations

import numpy as np

from .figure_detector import DetectedRegion

# TokenType index for TABLE (must match pdf_token_type_labels.TokenType)
_TOKEN_TYPE_TABLE: int = 3
_TOKEN_TYPE_PICTURE: int = 4


class RegionClassifier:
    """Assigns a human-readable type to each DetectedRegion."""

    # Thresholds (tunable)
    HIGH_COLOR_VARIANCE: float = 500.0   # lower for grayscale images
    HIGH_EDGE_DENSITY: float = 0.03      # lower for grayscale/background images
    VERY_HIGH_EDGE_DENSITY: float = 0.10

    def classify(self, region: DetectedRegion) -> str:
        """Return the best-guess region type.

        Order matters: table check comes first (strongest signal).
        Preserve pre-assigned types (e.g., ScannedPage from PyMuPDF).
        """
        # Preserve pre-assigned types from PyMuPDF detection
        if region.region_type in ("ScannedPage", "Figure"):
            return region.region_type

        # Table: strong horizontal AND vertical lines
        if region.is_table_like:
            return "Table"

        # Photo: high color variance, moderate-to-high edge density
        if (
            region.color_variance > self.HIGH_COLOR_VARIANCE
            and region.edge_density > self.HIGH_EDGE_DENSITY
        ):
            return "Photo"

        # Chart: low color variance, high edge density, sparse fill
        if (
            region.edge_density > self.HIGH_EDGE_DENSITY
            and region.fill_density < 0.4
        ):
            return "Chart"

        # Chart: very high edge density regardless of color
        if region.edge_density > self.VERY_HIGH_EDGE_DENSITY:
            return "Chart"

        # Logo: low color variance, low edge density, often wide
        if (
            region.color_variance <= self.HIGH_COLOR_VARIANCE
            and region.edge_density <= self.HIGH_EDGE_DENSITY
            and region.aspect_ratio > 1.0
        ):
            return "Logo"

        # Formula: tall, low color variance, moderate edge density
        if (
            region.aspect_ratio < 0.8
            and region.color_variance <= self.HIGH_COLOR_VARIANCE
            and region.edge_density > 0.03
        ):
            return "Formula"

        # Default: generic figure/image
        return "Figure"


def classify_regions(
    regions: list[DetectedRegion],
    page_tokens: list | None = None,
) -> list[DetectedRegion]:
    """Classify all regions in-place and return the list.

    Args:
        regions: Detected regions to classify.
        page_tokens: Optional list of PdfToken objects (with .token_type
            already set). If provided, region types are overridden based
            on the dominant token type overlapping each region.

    Returns:
        The same list of regions (modified in-place).
    """
    classifier = RegionClassifier()
    for region in regions:
        region.region_type = classifier.classify(region)

    if page_tokens:
        override_regions_by_token_types(regions, page_tokens)

    return regions


def override_regions_by_token_types(
    regions: list[DetectedRegion],
    page_tokens: list,
    min_table_ratio: float = 0.15,
) -> None:
    """Override region types based on the token types that fall inside them.

    When a region contains enough TABLE-type tokens (and few or no
    PICTURE-type tokens), reclassify it as "Table" — even if the
    image-based classifier thought it was a "Photo" or "Figure".

    Similarly, if a region contains many PICTURE-type tokens, reinforce
    "Figure"/"Photo" classification.

    This is useful when:
      - Tables lack visible grid lines (misclassified as Photo)
      - A table has a colored/dark background (high color variance → Photo)
      - Figures with dense edges are misclassified as Chart/Table

    Args:
        regions: Regions to override (modified in-place).
        page_tokens: List of tokens with .token_type set (as TokenType
            enum or int index).
        min_table_ratio: Minimum fraction of overlapping tokens that must
            be TABLE-type for a region to be overridden to "Table".
    """
    if not regions or not page_tokens:
        return

    for region in regions:
        # Find tokens that overlap this region
        overlapping_tokens = _tokens_in_region(region, page_tokens)
        if not overlapping_tokens:
            continue

        total = len(overlapping_tokens)
        table_count = sum(
            1 for t in overlapping_tokens
            if _token_type_index(t) == _TOKEN_TYPE_TABLE
        )
        picture_count = sum(
            1 for t in overlapping_tokens
            if _token_type_index(t) == _TOKEN_TYPE_PICTURE
        )

        table_ratio = table_count / total if total > 0 else 0
        picture_ratio = picture_count / total if total > 0 else 0

        # If a meaningful fraction of tokens are TABLE-type, override to Table
        # (but only if the region isn't already a known non-table type that
        #  has been explicitly confirmed)
        if (
            table_ratio >= min_table_ratio
            and table_ratio > picture_ratio
            and region.region_type not in ("ScannedPage", "Formula")
        ):
            region.region_type = "Table"

        # If mostly PICTURE tokens and region was misclassified as Table/Chart
        if (
            picture_ratio >= min_table_ratio
            and picture_ratio > table_ratio
            and region.region_type in ("Table", "Chart", "Logo")
        ):
            region.region_type = "Figure"


def _token_type_index(token) -> int:
    """Extract token type as an integer index from a PdfToken.

    Checks, in order:
      1. token.prediction (int, set by TokenTypeTrainer.predict())
      2. token.token_type (TokenType enum or int)
      3. Falls back to TEXT (6)

    Note: token.prediction is checked FIRST because token.token_type
    is always set to TokenType.TEXT by default, even before prediction.
    """
    # After trainer.predict(), the predicted type is in .prediction
    pred = getattr(token, 'prediction', None)
    if isinstance(pred, (int, np.integer)):
        return int(pred)

    # Check token_type (may be a TokenType enum from labeled data)
    tt = getattr(token, 'token_type', None)
    if tt is not None and hasattr(tt, 'get_index'):
        return tt.get_index()
    if isinstance(tt, (int, np.integer)):
        return int(tt)

    return 6  # default: TEXT


def _tokens_in_region(region: DetectedRegion, page_tokens: list) -> list:
    """Return tokens whose bounding box overlaps the region."""
    result = []
    for token in page_tokens:
        if token.page_number != region.page_number:
            continue
        bb = token.bounding_box
        if _rects_intersect(
            bb.left, bb.top, bb.right, bb.bottom,
            region.bbox.left, region.bbox.top,
            region.bbox.right, region.bbox.bottom,
        ):
            result.append(token)
    return result


def _rects_intersect(
    a_left: float, a_top: float, a_right: float, a_bottom: float,
    b_left: float, b_top: float, b_right: float, b_bottom: float,
) -> bool:
    """True if two rectangles intersect."""
    return not (
        a_right < b_left
        or a_left > b_right
        or a_bottom < b_top
        or a_top > b_bottom
    )
