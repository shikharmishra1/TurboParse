"""pdf_image_features — OpenCV-based figure/table detection for PDF documents.

Provides a hybrid pipeline:
  1. pypdfium2 renders each page to an image
  2. OpenCV processes the image to find non-text regions
  3. Heuristics classify regions (Photo, Chart, Logo, Formula, Table, Figure)
  4. Results are merged with the existing PdfFeatures token classifier

Primary API:
    pdf_features_with_images(pdf_path) → (PdfFeatures, token_image_overlaps dict)
    detect_figures_only(pdf_path) → list[DetectedRegion]
"""

from .integration import pdf_features_with_images, detect_figures_only
from .figure_detector import FigureDetector, DetectedRegion, PageRegions
from .region_classifier import (
    RegionClassifier,
    classify_regions,
    override_regions_by_token_types,
)

__all__ = [
    "pdf_features_with_images",
    "detect_figures_only",
    "FigureDetector",
    "DetectedRegion",
    "PageRegions",
    "RegionClassifier",
    "classify_regions",
    "override_regions_by_token_types",
]
