"""Figure and table detection using OpenCV image processing.

Pipeline:
  1. Render PDF page → grayscale image
  2. Adaptive threshold → binary (text becomes white blobs on black)
  3. Morphological opening to remove text (small, dense components)
  4. Subtract text mask → candidate non-text regions
  5. Find connected components → region candidates
  6. Filter by size, aspect ratio, density
  7. Reject tables by detecting grid lines
  8. Optionally: create a mask from known token bounding boxes and subtract
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from pdf_features.Rectangle import Rectangle


@dataclass
class DetectedRegion:
    """A candidate non-text region detected on a page."""

    bbox: Rectangle
    page_number: int
    area: int
    width: int
    height: int
    aspect_ratio: float       # width / height
    fill_density: float       # fraction of pixels that are foreground
    edge_density: float        # fraction of edge pixels
    color_variance: float      # variance of grayscale values in original image
    horizontal_line_count: int  # number of long horizontal lines
    vertical_line_count: int    # number of long vertical lines
    region_type: str = "unknown"  # assigned by RegionClassifier

    @property
    def is_table_like(self) -> bool:
        """A region is table-like if it has many horizontal/vertical lines
        AND is not a solid block (which would give false line counts)."""
        if self.fill_density > 0.90:
            return False  # solid region — not a table
        return self.horizontal_line_count >= 3 and self.vertical_line_count >= 3


@dataclass
class PageRegions:
    """All detected non-text regions for a single page."""

    page_number: int
    page_width: int
    page_height: int
    regions: list[DetectedRegion] = field(default_factory=list)

    def iter_regions(self):
        yield from self.regions


class FigureDetector:
    """Detects figures, images, and table regions in rendered PDF pages.

    All kernel sizes are defined at a reference DPI of 150 and are
    scaled proportionally for the actual render DPI.
    """

    # ── reference DPI for kernel scaling ──────────────────────────────
    REFERENCE_DPI: int = 150

    # ── threshold parameters ──────────────────────────────────────────
    ADAPTIVE_BLOCK_SIZE: int = 31     # must be odd; scaled by DPI
    ADAPTIVE_C: int = 15              # scaled by DPI

    # ── text-removal morphology (sizes at REFERENCE_DPI) ──────────────
    # Closing (dilate → erode): merges nearby text into solid blocks,
    # creating a text mask.  Subtract from binary to reveal non-text.
    # Works best on born-digital PDFs where text is rendered as thin
    # strokes.  For scanned/image PDFs, prefer detect_with_token_mask().
    TEXT_CLOSE_KERNEL_WIDTH: int = 50
    TEXT_CLOSE_KERNEL_HEIGHT: int = 10

    # ── filter thresholds (in pixels at REFERENCE_DPI) ────────────────
    MIN_AREA: int = 2000              # minimum connected-component area (px²)
    MIN_WIDTH: int = 60               # minimum region width (px)
    MIN_HEIGHT: int = 40              # minimum region height (px)
    MIN_ASPECT_RATIO: float = 0.05    # minimum width/height
    MAX_ASPECT_RATIO: float = 20.0    # maximum width/height
    MIN_FILL_DENSITY: float = 0.02    # at least 2% foreground pixels after subtract
    MAX_FILL_DENSITY: float = 0.90    # at most 90% foreground pixels

    # ── table-line detection (sizes at REFERENCE_DPI) ─────────────────
    H_LINE_KERNEL_WIDTH: int = 60
    H_LINE_KERNEL_HEIGHT: int = 1
    V_LINE_KERNEL_WIDTH: int = 1
    V_LINE_KERNEL_HEIGHT: int = 60
    MIN_LINE_LENGTH_RATIO: float = 0.35

    # ── grayscale detection (for scanned/background-image PDFs) ───────
    GRAY_THRESHOLD: int = 180         # dark-enough to exclude light backgrounds
    GRAY_MIN_FILL: float = 0.15       # at least 15% of region is dark content
    GRAY_MIN_EDGE_DENSITY: float = 0.02  # min edge density for a "real" figure
    GRAY_BORDER_MARGIN: int = 50      # px from page edge to ignore (margins)
    GRAY_MAX_PAGE_FRACTION: float = 0.70  # ignore regions covering >70% of page

    # ── token-mask dilation (in px at REFERENCE_DPI) ──────────────────
    TOKEN_MASK_DILATION: int = 3

    # ── skip grid-line detection below this area (REFERENCE_DPI px²) ──
    MIN_GRID_AREA: int = 5000

    # ── internal state ────────────────────────────────────────────────
    _gray: Optional[np.ndarray] = None
    _binary: Optional[np.ndarray] = None
    _original_rgb: Optional[np.ndarray] = None
    _scale: float = 1.0  # dpi / REFERENCE_DPI

    # ── cached kernels (pre-computed in _set_scale) ───────────────────
    _h_line_kernel: Optional[np.ndarray] = None
    _v_line_kernel: Optional[np.ndarray] = None
    _dilate_kernel: Optional[np.ndarray] = None
    _close_kernel: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _set_scale(self, dpi: int) -> None:
        """Compute scale factor relative to reference DPI and pre-build kernels."""
        self._scale = dpi / self.REFERENCE_DPI

        # Pre-compute line-detection kernels (created once per page, not per region)
        self._h_line_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self._s(self.H_LINE_KERNEL_WIDTH), self._s(self.H_LINE_KERNEL_HEIGHT)),
        )
        self._v_line_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self._s(self.V_LINE_KERNEL_WIDTH), self._s(self.V_LINE_KERNEL_HEIGHT)),
        )

        # Pre-compute dilation kernel for token masks
        dilate_px = max(1, self._s(self.TOKEN_MASK_DILATION))
        self._dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_px, dilate_px)
        )

        # Pre-compute tiny close kernel (3×3, fixed size)
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def _s(self, value: int) -> int:
        """Scale an integer parameter by the current DPI ratio."""
        return max(1, int(value * self._scale))

    def detect(
        self,
        page_image: np.ndarray,
        page_number: int,
        page_width: int,
        page_height: int,
        dpi: int = 200,
    ) -> PageRegions:
        """Run morphological detection pipeline (best for born-digital PDFs).

        Pipeline:
          1. Adaptive threshold → inverted binary
          2. Morphological closing → text mask
          3. Subtract text mask → candidate non-text regions
          4. Connected components → region candidates
          5. Filter by size, density, aspect ratio

        Accepts either RGB (3-channel) or grayscale (2D) input.
        """
        self._set_scale(dpi)

        if page_image.ndim == 2:
            self._gray = page_image
            self._original_rgb = None
        else:
            self._original_rgb = page_image
            self._gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)

        self._binary = self._threshold(self._gray)

        text_mask = self._create_text_mask(self._binary)
        candidate = cv2.subtract(self._binary, text_mask)

        regions = self._extract_regions(
            candidate, page_number, page_width, page_height, dpi
        )

        return PageRegions(
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            regions=regions,
        )

    def detect_with_token_mask(
        self,
        page_image: np.ndarray,
        page_number: int,
        page_width: int,
        page_height: int,
        token_boxes: list[Rectangle],
        dpi: int = 200,
    ) -> PageRegions:
        """Alternative pipeline: use known token boxes to subtract text.

        Accepts either RGB (3-channel) or grayscale (2D) input.
        """
        self._set_scale(dpi)

        if page_image.ndim == 2:
            self._gray = page_image
            self._original_rgb = None
        else:
            self._original_rgb = page_image
            self._gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
        self._binary = self._threshold(self._gray)

        scale = dpi / 72.0  # PDF points → pixels
        h, w = self._gray.shape
        token_mask = self._build_token_mask_fast(token_boxes, scale, h, w)

        candidate = cv2.subtract(self._binary, token_mask)

        regions = self._extract_regions(
            candidate, page_number, page_width, page_height, dpi
        )
        return PageRegions(
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            regions=regions,
        )

    def detect_grayscale_with_token_mask(
        self,
        page_image: np.ndarray,
        page_number: int,
        page_width: int,
        page_height: int,
        token_boxes: list[Rectangle],
        dpi: int = 200,
    ) -> PageRegions:
        """Pipeline for pages with text overlaid on a background image.

        Strategy:
        1. Mask known text regions → white.
        2. Threshold at GRAY_THRESHOLD (lower = only dark content survives).
        3. Light morphological close to bridge small gaps only.
        4. Connected components → filter by size, fill, edge density.
        5. Exclude margin regions touching page borders.
        6. Reject regions covering >GRAY_MAX_PAGE_FRACTION of the page
           (these are full-page backgrounds, not distinct figures).

        Accepts either RGB (3-channel) or grayscale (2D) input.
        """
        self._set_scale(dpi)

        if page_image.ndim == 2:
            self._gray = page_image
            self._original_rgb = None
        else:
            self._original_rgb = page_image
            self._gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)

        scale = dpi / 72.0
        h, w = self._gray.shape
        border = self._s(self.GRAY_BORDER_MARGIN)
        page_px_area = h * w

        # ── Build and dilate token mask ────────────────────────────────
        token_mask = self._build_token_mask_fast(token_boxes, scale, h, w)
        # Extra dilation iteration for grayscale mode (text may bleed into background)
        token_mask = cv2.dilate(token_mask, self._dilate_kernel, iterations=1)

        # ── Remove text, threshold to find dark content ────────────────
        gray_no_text = self._gray.copy()
        gray_no_text[token_mask > 0] = 255

        # Lower threshold: only significantly dark pixels are "content"
        _, candidate = cv2.threshold(
            gray_no_text, self.GRAY_THRESHOLD, 255, cv2.THRESH_BINARY_INV
        )

        # ── Exclude page border region (margins) ──────────────────────
        candidate[:border, :] = 0
        candidate[-border:, :] = 0
        candidate[:, :border] = 0
        candidate[:, -border:] = 0

        # ── Light morphological close (small gaps only) ───────────────
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, self._close_kernel)

        self._binary = candidate

        # ── Extract and filter regions ─────────────────────────────────
        regions = self._extract_regions_grayscale(
            candidate, gray_no_text, page_number, page_width, page_height, dpi
        )

        # Reject regions that cover most of the page (full-page backgrounds)
        regions = [
            r for r in regions
            if r.area / max(page_px_area, 1) < self.GRAY_MAX_PAGE_FRACTION
        ]

        return PageRegions(
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            regions=regions,
        )

    def detect_dark_blobs(
        self,
        page_image: np.ndarray,
        page_number: int,
        page_width: int,
        page_height: int,
        token_boxes: list[Rectangle],
        dpi: int = 200,
    ) -> PageRegions:
        """Find dark, non-text blobs — best for distinct figures on a
        lighter background.

        Instead of removing text first (which merges everything), this:
        1. Thresholds the ORIGINAL grayscale to find dark blobs.
        2. Removes dark pixels that overlap token boxes (text).
        3. What's left: dark regions that are NOT text → figures/photos.

        Accepts either RGB (3-channel) or grayscale (2D) input.
        When already grayscale, the cvtColor step is skipped.
        """
        self._set_scale(dpi)

        # Auto-detect: if input is already grayscale (2D), use it directly
        if page_image.ndim == 2:
            self._gray = page_image
            self._original_rgb = None  # not available; _extract_regions_grayscale doesn't use it
        else:
            self._original_rgb = page_image
            self._gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)

        scale = dpi / 72.0
        h, w = self._gray.shape
        border = self._s(self.GRAY_BORDER_MARGIN)
        page_px_area = h * w

        # ── Threshold original grayscale to find dark content ─────────
        _, dark = cv2.threshold(
            self._gray, self.GRAY_THRESHOLD, 255, cv2.THRESH_BINARY_INV
        )

        # ── Build token mask ──────────────────────────────────────────
        token_mask = self._build_token_mask_fast(token_boxes, scale, h, w)

        # ── Exclude page borders ──────────────────────────────────────
        dark[:border, :] = 0
        dark[-border:, :] = 0
        dark[:, :border] = 0
        dark[:, -border:] = 0

        # ── Remove dark pixels that overlap text ──────────────────────
        dark[token_mask > 0] = 0

        # ── Light close to bridge tiny gaps (1-2px) ───────────────────
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, self._close_kernel)

        self._binary = dark

        # ── Extract and filter ────────────────────────────────────────
        regions = self._extract_regions_grayscale(
            dark, self._gray, page_number, page_width, page_height, dpi
        )

        # Reject full-page backgrounds
        regions = [
            r for r in regions
            if r.area / max(page_px_area, 1) < self.GRAY_MAX_PAGE_FRACTION
        ]

        return PageRegions(
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            regions=regions,
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _build_token_mask_fast(
        self,
        token_boxes: list[Rectangle],
        scale: float,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Build a token mask from bounding boxes (vectorized).

        Returns a dilated mask where token regions are 255.
        """
        token_mask = np.zeros((h, w), dtype=np.uint8)
        for box in token_boxes:
            x1 = max(0, int(box.left * scale))
            y1 = max(0, int(box.top * scale))
            x2 = min(w, int(box.right * scale))
            y2 = min(h, int(box.bottom * scale))
            if x2 > x1 and y2 > y1:
                token_mask[y1:y2, x1:x2] = 255
        if token_boxes:
            token_mask = cv2.dilate(token_mask, self._dilate_kernel, iterations=1)
        return token_mask

    def _threshold(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive threshold → inverted binary (foreground=255)."""
        block = self._s(self.ADAPTIVE_BLOCK_SIZE)
        if block % 2 == 0:
            block += 1  # must be odd
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            self._s(self.ADAPTIVE_C),
        )

    def _create_text_mask(self, binary: np.ndarray) -> np.ndarray:
        """Build a text mask via morphological closing.

        Closing (dilate → erode) merges nearby text strokes into solid
        blocks.  Subtract this mask from the binary to reveal non-text
        regions (figures, images, charts).

        NOTE: This works well on born-digital PDFs where text is thin.
        For scanned/image-based PDFs, use detect_with_token_mask().
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                self._s(self.TEXT_CLOSE_KERNEL_WIDTH),
                self._s(self.TEXT_CLOSE_KERNEL_HEIGHT),
            ),
        )
        dilated = cv2.dilate(binary, kernel, iterations=1)
        text_mask = cv2.erode(dilated, kernel, iterations=1)
        return text_mask

    def _extract_regions(
        self,
        candidate: np.ndarray,
        page_number: int,
        page_width: int,
        page_height: int,
        dpi: int,
    ) -> list[DetectedRegion]:
        """Find connected components, filter, and build DetectedRegion list."""
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            candidate, connectivity=8
        )

        scale = dpi / 72.0  # pixels → PDF points
        regions: list[DetectedRegion] = []

        # Scaled minimum area for skipping grid-line detection
        min_grid_area = self._s(self.MIN_GRID_AREA)

        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]

            if not self._passes_size_filter(area, w, h):
                continue

            aspect = w / h if h > 0 else 0
            if not (self.MIN_ASPECT_RATIO <= aspect <= self.MAX_ASPECT_RATIO):
                continue

            roi_binary = candidate[y : y + h, x : x + w]
            roi_gray = self._gray[y : y + h, x : x + w]

            fill = float(cv2.countNonZero(roi_binary)) / max(area, 1)
            if fill < self.MIN_FILL_DENSITY or fill > self.MAX_FILL_DENSITY:
                continue

            edge_den = self._compute_edge_density(roi_gray)
            color_var = float(np.var(roi_gray))

            # Only detect grid lines for regions large enough to be tables
            if area >= min_grid_area:
                h_lines, v_lines = self._count_grid_lines(roi_binary, w, h)
            else:
                h_lines, v_lines = 0, 0

            bbox = Rectangle.from_width_height(
                left=int(x / scale),
                top=int(y / scale),
                width=int(w / scale),
                height=int(h / scale),
            )

            regions.append(
                DetectedRegion(
                    bbox=bbox,
                    page_number=page_number,
                    area=int(area),
                    width=int(w / scale),
                    height=int(h / scale),
                    aspect_ratio=aspect,
                    fill_density=fill,
                    edge_density=edge_den,
                    color_variance=color_var,
                    horizontal_line_count=h_lines,
                    vertical_line_count=v_lines,
                )
            )

        return regions

    def _passes_size_filter(self, area: int, w: int, h: int) -> bool:
        return (
            area >= self._s(self.MIN_AREA)
            and w >= self._s(self.MIN_WIDTH)
            and h >= self._s(self.MIN_HEIGHT)
        )

    def _extract_regions_grayscale(
        self,
        candidate: np.ndarray,
        gray_no_text: np.ndarray,
        page_number: int,
        page_width: int,
        page_height: int,
        dpi: int,
    ) -> list[DetectedRegion]:
        """Extract regions from grayscale token-mask subtraction.

        Fill density = fraction of pixels that are non-white (< GRAY_THRESHOLD).
        Also requires minimum edge density to filter out blank background.
        """
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            candidate, connectivity=8
        )

        scale = dpi / 72.0
        regions: list[DetectedRegion] = []
        threshold = self.GRAY_THRESHOLD

        # Scaled minimum area for skipping grid-line detection
        min_grid_area = self._s(self.MIN_GRID_AREA)

        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]

            if not self._passes_size_filter(area, w, h):
                continue

            aspect = w / h if h > 0 else 0
            if not (self.MIN_ASPECT_RATIO <= aspect <= self.MAX_ASPECT_RATIO):
                continue

            roi_gray = gray_no_text[y : y + h, x : x + w]

            # Fill density: fraction of pixels that are non-white content
            non_white = np.count_nonzero(roi_gray < threshold)
            fill = float(non_white) / max(area, 1)
            if fill < self.GRAY_MIN_FILL:
                continue

            edge_den = self._compute_edge_density(roi_gray)
            if edge_den < self.GRAY_MIN_EDGE_DENSITY:
                continue

            color_var = float(np.var(roi_gray))

            # Only detect grid lines for regions large enough to be tables
            if area >= min_grid_area:
                roi_binary = candidate[y : y + h, x : x + w]
                h_lines, v_lines = self._count_grid_lines(roi_binary, w, h)
            else:
                h_lines, v_lines = 0, 0

            bbox = Rectangle.from_width_height(
                left=int(x / scale),
                top=int(y / scale),
                width=int(w / scale),
                height=int(h / scale),
            )

            regions.append(
                DetectedRegion(
                    bbox=bbox,
                    page_number=page_number,
                    area=int(area),
                    width=int(w / scale),
                    height=int(h / scale),
                    aspect_ratio=aspect,
                    fill_density=fill,
                    edge_density=edge_den,
                    color_variance=color_var,
                    horizontal_line_count=h_lines,
                    vertical_line_count=v_lines,
                )
            )

        return regions

    # ------------------------------------------------------------------
    # Feature computation helpers
    # ------------------------------------------------------------------

    def _compute_edge_density(self, gray_roi: np.ndarray) -> float:
        """Fraction of edge pixels (Canny) in the region."""
        if gray_roi.size == 0:
            return 0.0
        edges = cv2.Canny(gray_roi, 50, 150)
        return float(cv2.countNonZero(edges)) / gray_roi.size

    def _count_grid_lines(
        self, binary_roi: np.ndarray, w: int, h: int
    ) -> tuple[int, int]:
        """Count long horizontal and vertical lines in a binary region.

        Uses kernels pre-computed in _set_scale() — no allocation per call.
        """
        h_lines_img = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, self._h_line_kernel)
        h_line_count = self._count_long_lines(h_lines_img, w, "horizontal")

        v_lines_img = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, self._v_line_kernel)
        v_line_count = self._count_long_lines(v_lines_img, h, "vertical")

        return h_line_count, v_line_count

    def _count_long_lines(
        self, line_img: np.ndarray, region_extent: int, orientation: str
    ) -> int:
        """Count lines that span at least MIN_LINE_LENGTH_RATIO of the region."""
        min_length = int(region_extent * self.MIN_LINE_LENGTH_RATIO)
        if min_length < 1:
            return 0

        # Projection: sum along the axis orthogonal to the line direction
        if orientation == "horizontal":
            projection = np.sum(line_img, axis=1)  # sum each row
        else:
            projection = np.sum(line_img, axis=0)  # sum each column

        # A "long line" is a row/col whose projection exceeds min_length*255
        return int(np.sum(projection >= min_length * 255))
