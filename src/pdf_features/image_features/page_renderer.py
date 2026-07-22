"""Render PDF pages to numpy arrays using pypdfium2.

Provides both RGB and grayscale rendering at configurable DPI.
"""

from __future__ import annotations

import cv2
import numpy as np
import pypdfium2 as pdfium


def render_page_to_image(
    doc: pdfium.PdfDocument, page_index: int, dpi: int = 200
) -> np.ndarray:
    """Render a single PDF page to an RGB numpy array.

    Args:
        doc: A pypdfium2.PdfDocument.
        page_index: Zero-based page number.
        dpi: Render resolution in dots per inch.

    Returns:
        An RGB image as a numpy array of shape (height, width, 3).
    """
    page = doc[page_index]
    scale = dpi / 72.0  # PDF points → pixels
    bitmap = page.render(scale=scale)
    img = bitmap.to_numpy()
    # pypdfium2 returns BGRA; convert to RGB
    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    return img


def render_page_to_gray(
    doc: pdfium.PdfDocument, page_index: int, dpi: int = 200
) -> np.ndarray:
    """Render a single PDF page to a grayscale numpy array.

    Uses the pypdfium2 bitmap buffer directly to avoid an intermediate
    BGRA → BGR copy before the grayscale conversion.

    Args:
        doc: A pypdfium2.PdfDocument.
        page_index: Zero-based page number.
        dpi: Render resolution in dots per inch.

    Returns:
        A grayscale image as a numpy array of shape (height, width).
    """
    page = doc[page_index]
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    # Convert BGRA → grayscale in one step (skip BGRA→BGR copy)
    gray = cv2.cvtColor(bitmap.to_numpy(), cv2.COLOR_BGRA2GRAY)
    return gray


def get_page_dimensions_at_dpi(
    doc: pdfium.PdfDocument, page_index: int, dpi: int = 200
) -> tuple[int, int]:
    """Return (width, height) of a rendered page at the given DPI."""
    page = doc[page_index]
    w_pt, h_pt = page.get_size()
    scale = dpi / 72.0
    return int(w_pt * scale), int(h_pt * scale)

