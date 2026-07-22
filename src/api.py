"""
FastAPI application for TurboParse PDF parsing.

Routes:
    POST /parse          — Upload a PDF, get unified predictions + regions + TSR.
    GET  /health         — Health check.

Query parameters (all optional):
    include_markdown     — Include markdown in `conversion_result` field (default: false)
    include_image_data   — Include base64 image crops for picture regions (default: false)
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pypdfium2 as pdfium
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Ensure src/ is on sys.path ────────────────────────────────────────
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import predict_with_layout, pdf_features_with_images
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from converters.markdown_converter import predictions_to_markdown
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from pdf_features.image_features.page_renderer import render_page_to_image

logger = logging.getLogger(__name__)


# ── Lifespan: load model once at startup ──────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model path on startup. Cleanup on shutdown."""
    # Nothing to pre-load; model is loaded per-request by TokenTypeTrainer
    logger.info("TurboParse API started")
    yield


app = FastAPI(
    title="TurboParse API",
    description="Parse PDFs into structured predictions with layout analysis and TSR.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Default model path (relative to workspace root) ───────────────────
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "pdf_tokens_type.model"


# ── Pydantic schemas ─────────────────────────────────────────────────


class BoundingBox(BaseModel):
    left: int
    top: int
    right: int
    bottom: int


class TokenOut(BaseModel):
    text: str
    token_type: str
    bounding_box: BoundingBox


class TSRCell(BaseModel):
    row: int
    col: int
    bounding_box: BoundingBox
    text: str = ""


class TSRResult(BaseModel):
    num_cells: int
    cells: list[TSRCell]


class RegionOut(BaseModel):
    region_type: str
    bounding_box: BoundingBox
    width: int
    height: int
    area: int
    aspect_ratio: float
    # Only included when include_image_data=True
    image_base64: Optional[str] = None
    # Only for Table regions — nested TSR result
    tsr_result: Optional[TSRResult] = None


class PageOut(BaseModel):
    page_number: int
    width: int
    height: int
    tokens: list[TokenOut]
    regions: list[RegionOut]


class ParseResponse(BaseModel):
    pages: list[PageOut]
    conversion_result: Optional[str] = None  # only when include_markdown=True
    num_pages: int
    num_tokens: int
    num_regions: int


class HealthResponse(BaseModel):
    status: str
    model_available: bool


# ── Helpers ───────────────────────────────────────────────────────────


def _bbox_to_dict(bbox) -> BoundingBox:
    """Convert a Rectangle (or dict with left/top/right/bottom) to BoundingBox."""
    if isinstance(bbox, dict):
        return BoundingBox(
            left=int(bbox.get("left", 0)),
            top=int(bbox.get("top", 0)),
            right=int(bbox.get("right", 0)),
            bottom=int(bbox.get("bottom", 0)),
        )
    return BoundingBox(left=bbox.left, top=bbox.top, right=bbox.right, bottom=bbox.bottom)


def _find_text_in_cell(
    cell_bbox_pdf: dict,
    page_tokens: list,
) -> str:
    """Find text tokens that overlap with a TSR cell bounding box (in PDF coords)."""
    cx1 = cell_bbox_pdf["x"]
    cy1 = cell_bbox_pdf["y"]
    cx2 = cx1 + cell_bbox_pdf["w"]
    cy2 = cy1 + cell_bbox_pdf["h"]

    cell_texts: list[str] = []
    for token in page_tokens:
        bb = token.bounding_box
        # Check overlap
        if bb.right <= cx1 or bb.left >= cx2 or bb.bottom <= cy1 or bb.top >= cy2:
            continue
        cell_texts.append(token.content)

    return " ".join(cell_texts)


def _associate_tsr_with_regions(
    table_tsr_results: list[dict],
    regions: list,
    pdf_features: PdfFeatures,
) -> dict[tuple[int, int, int, int], dict]:
    """Match TSR results to Table regions by page + bounding box proximity.

    Returns a dict keyed by (page, left, top, right, bottom) → tsr_dict.
    """
    tsr_map: dict[tuple[int, int, int, int], dict] = {}

    # Build page token index for text lookup
    tokens_by_page: dict[int, list] = {}
    for page in pdf_features.pages:
        tokens_by_page[page.page_number] = list(page.tokens)

    for tsr in table_tsr_results:
        tsr_page = tsr["page"]
        tsr_bbox = tsr["bbox"]
        tsr_key = (tsr_page, tsr_bbox["left"], tsr_bbox["top"],
                    tsr_bbox["right"], tsr_bbox["bottom"])

        # Build enriched cells with text
        page_tokens = tokens_by_page.get(tsr_page, [])
        cells_out: list[TSRCell] = []
        for cell in tsr.get("cells", []):
            cell_bbox = BoundingBox(
                left=int(cell["x"]),
                top=int(cell["y"]),
                right=int(cell["x"] + cell["w"]),
                bottom=int(cell["y"] + cell["h"]),
            )
            cell_text = _find_text_in_cell(cell, page_tokens)
            cells_out.append(TSRCell(
                row=cell.get("row", 0),
                col=cell.get("col", 0),
                bounding_box=cell_bbox,
                text=cell_text,
            ))

        tsr_map[tsr_key] = {
            "num_cells": tsr.get("num_cells", 0),
            "cells": cells_out,
        }

    return tsr_map


# ── Routes ────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check if the API and model are available."""
    return HealthResponse(
        status="ok",
        model_available=DEFAULT_MODEL_PATH.exists(),
    )


@app.post("/parse", response_model=ParseResponse)
async def parse_pdf(
    file: UploadFile = File(..., description="PDF file to parse"),
    include_markdown: bool = Query(False, description="Include markdown conversion in response"),
    include_image_data: bool = Query(False, description="Include base64-encoded image crops for picture regions"),
    model_path: str = Query("", description="Optional path to custom model file"),
    dpi: int = Query(150, ge=72, le=600, description="Render DPI for image processing"),
):
    """Parse a PDF and return structured predictions with layout analysis.

    Returns tokens, detected regions (tables/figures), and TSR table
    structure — all in a unified per-page schema.  TSR results are nested
    under their parent Table region.
    """
    # ── Save uploaded file to temp location ───────────────────────────
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    tmp_path = Path(tempfile.gettempdir()) / f"turboparse_{uuid.uuid4().hex}{suffix}"

    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        # ── Determine model path ──────────────────────────────────────
        mp = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        mp_str = str(mp) if mp.exists() else None

        # ── Run the full pipeline ─────────────────────────────────────
        pdf_doc = pdfium.PdfDocument(str(tmp_path))

        pdf_features, token_image_overlaps, all_regions = pdf_features_with_images(
            str(tmp_path), dpi=dpi, pdf_document=pdf_doc,
        )

        pdf_features, predictions, all_figures, table_tsr_results = predict_with_layout(
            str(tmp_path),
            pdf_features=pdf_features,
            token_image_overlaps=token_image_overlaps,
            all_regions=all_regions,
            model_path=mp_str,
            dpi=dpi,
            pdf_document=pdf_doc,
        )

        # ── Associate TSR results with Table regions ──────────────────
        tsr_map = _associate_tsr_with_regions(
            table_tsr_results, all_figures, pdf_features,
        )

        # ── Build per-page output ─────────────────────────────────────
        pages_out: list[PageOut] = []

        for page in pdf_features.pages:
            pn = page.page_number

            # Tokens for this page
            page_tokens: list[TokenOut] = []
            for p in predictions:
                if p["page_number"] != pn:
                    continue
                bb = p.get("bounding_box", {})
                page_tokens.append(TokenOut(
                    text=p.get("text_content", ""),
                    token_type=p.get("token_type", "Text"),
                    bounding_box=BoundingBox(
                        left=bb.get("left", 0),
                        top=bb.get("top", 0),
                        right=bb.get("right", 0),
                        bottom=bb.get("bottom", 0),
                    ),
                ))

            # Regions for this page
            page_regions: list[RegionOut] = []
            for region in all_figures:
                if region.page_number != pn:
                    continue

                region_bbox = _bbox_to_dict(region.bbox)

                # Check for TSR match (Table regions only)
                tsr_out: Optional[TSRResult] = None
                if region.region_type == "Table":
                    tsr_key = (pn, region.bbox.left, region.bbox.top,
                               region.bbox.right, region.bbox.bottom)
                    tsr_data = tsr_map.get(tsr_key)
                    if tsr_data:
                        tsr_out = TSRResult(
                            num_cells=tsr_data["num_cells"],
                            cells=tsr_data["cells"],
                        )

                # Image data (only if requested and applicable)
                image_b64: Optional[str] = None
                if include_image_data and region.region_type in ("Picture", "Photo", "Figure", "Chart"):
                    try:
                        pi = pn - 1
                        page_img = render_page_to_image(pdf_doc, pi, dpi=dpi)
                        scale = dpi / 72.0
                        x1 = int(region.bbox.left * scale)
                        y1 = int(region.bbox.top * scale)
                        x2 = int(region.bbox.right * scale)
                        y2 = int(region.bbox.bottom * scale)
                        h_img, w_img = page_img.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_img, x2), min(h_img, y2)
                        if x2 > x1 and y2 > y1:
                            crop = page_img[y1:y2, x1:x2]
                            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                            _, buf = cv2.imencode(".png", crop_bgr)
                            image_b64 = base64.b64encode(buf).decode("utf-8")
                    except Exception:
                        pass

                page_regions.append(RegionOut(
                    region_type=region.region_type,
                    bounding_box=region_bbox,
                    width=region.width,
                    height=region.height,
                    area=region.area,
                    aspect_ratio=round(region.aspect_ratio, 4),
                    image_base64=image_b64,
                    tsr_result=tsr_out,
                ))

            pages_out.append(PageOut(
                page_number=pn,
                width=page.page_width,
                height=page.page_height,
                tokens=page_tokens,
                regions=page_regions,
            ))

        # ── Markdown (only if requested) ──────────────────────────────
        md_text: Optional[str] = None
        if include_markdown:
            try:
                md_text = predictions_to_markdown(
                    pdf_features=pdf_features,
                    predictions=predictions,
                    regions=all_figures,
                    table_tsr_results=table_tsr_results,
                    pdf_path=str(tmp_path),
                    pdf_document=pdf_doc,
                )
            except Exception as e:
                logger.warning("Markdown conversion failed: %s", e)
                md_text = None

        pdf_doc.close()

        return ParseResponse(
            pages=pages_out,
            conversion_result=md_text,
            num_pages=len(pages_out),
            num_tokens=sum(len(p.tokens) for p in pages_out),
            num_regions=sum(len(p.regions) for p in pages_out),
        )

    except Exception as e:
        logger.exception("Parse failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "PDF parsing failed"},
        )
    finally:
        # Clean up temp file
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ── Run with: uvicorn api:app ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
