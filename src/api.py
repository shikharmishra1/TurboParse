"""
FastAPI application for TurboParse PDF parsing.

Routes:
    POST /parse          — Upload a PDF, get a ZIP with document.json + markdown.md + images/.
    GET  /health         — Health check.

The ZIP response contains:
    document.json        — Structured predictions (tokens, regions, TSR cells)
    markdown.md          — Markdown conversion with relative image paths
    images/              — Cropped region images as PNG files

Query parameters (all optional):
    include_markdown     — Include markdown.md in the ZIP (default: true)
    include_images       — Include images/ folder in the ZIP (default: true)
    dpi                  — Render DPI (default: 150)
    model_path           — Custom model path (default: auto-detect)
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import pypdfium2 as pdfium
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

# ── Ensure src/ is on sys.path ────────────────────────────────────────
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import predict_with_layout, pdf_features_with_images
from converters.markdown_converter import MarkdownConverter
from pdf_features.PdfFeatures import PdfFeatures

logger = logging.getLogger(__name__)


# ── Lifespan: load model once at startup ──────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nothing to pre-load; model is loaded per-request by TokenTypeTrainer."""
    logger.info("TurboParse API started")
    yield


app = FastAPI(
    title="TurboParse API",
    description="Parse PDFs into a ZIP with structured JSON + Markdown + images.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Model path: env var takes priority, then local fallback ───────────
DEFAULT_MODEL_PATH = Path(
    os.environ.get("MODEL_PATH")
    or Path(__file__).resolve().parent.parent / "model" / "pdf_tokens_type.model"
)


# ── Helpers ───────────────────────────────────────────────────────────


def _build_document_json(
    pdf_features: PdfFeatures,
    predictions: list[dict],
    all_figures: list,
    table_tsr_results: list[dict],
) -> dict:
    """Build the unified document.json structure with TSR nested under tables."""

    # ── Index TSR results by (page, left, top, right, bottom) ───────
    tsr_by_key: dict[tuple, dict] = {}
    tokens_by_page: dict[int, list] = {}
    for page in pdf_features.pages:
        tokens_by_page[page.page_number] = list(page.tokens)

    for tsr in table_tsr_results:
        bb = tsr["bbox"]
        key = (tsr["page"], bb["left"], bb["top"], bb["right"], bb["bottom"])
        tsr_by_key[key] = tsr

    # ── Build pages ─────────────────────────────────────────────────
    pages_out: list[dict] = []

    for page in pdf_features.pages:
        pn = page.page_number

        # Tokens
        page_tokens: list[dict] = []
        for p in predictions:
            if p["page_number"] != pn:
                continue
            bb = p.get("bounding_box", {})
            page_tokens.append({
                "text": p.get("text_content", ""),
                "token_type": p.get("token_type", "Text"),
                "bounding_box": {
                    "left": bb.get("left", 0),
                    "top": bb.get("top", 0),
                    "right": bb.get("right", 0),
                    "bottom": bb.get("bottom", 0),
                },
            })

        # Regions (with TSR nested under Table regions)
        page_regions: list[dict] = []
        page_tokens_list = tokens_by_page.get(pn, [])

        for region in all_figures:
            if region.page_number != pn:
                continue

            r = {
                "region_type": region.region_type,
                "bounding_box": {
                    "left": region.bbox.left,
                    "top": region.bbox.top,
                    "right": region.bbox.right,
                    "bottom": region.bbox.bottom,
                },
                "width": region.width,
                "height": region.height,
                "area": region.area,
                "aspect_ratio": round(region.aspect_ratio, 4),
            }

            # Nest TSR under Table regions
            if region.region_type == "Table":
                tsr_key = (pn, region.bbox.left, region.bbox.top,
                           region.bbox.right, region.bbox.bottom)
                tsr_data = tsr_by_key.get(tsr_key)
                if tsr_data:
                    cells_out: list[dict] = []
                    for cell in tsr_data.get("cells", []):
                        cx = cell["x"]
                        cy = cell["y"]
                        cw = cell["w"]
                        ch = cell["h"]
                        # Find text in this cell
                        cell_texts: list[str] = []
                        for tok in page_tokens_list:
                            bb = tok.bounding_box
                            if (bb.right > cx and bb.left < cx + cw
                                    and bb.bottom > cy and bb.top < cy + ch):
                                cell_texts.append(tok.content)
                        cells_out.append({
                            "row": cell.get("row", 0),
                            "col": cell.get("col", 0),
                            "bounding_box": {
                                "left": int(cx),
                                "top": int(cy),
                                "right": int(cx + cw),
                                "bottom": int(cy + ch),
                            },
                            "text": " ".join(cell_texts),
                        })
                    r["tsr_result"] = {
                        "num_cells": tsr_data.get("num_cells", 0),
                        "cells": cells_out,
                    }

            page_regions.append(r)

        pages_out.append({
            "page_number": pn,
            "width": page.page_width,
            "height": page.page_height,
            "tokens": page_tokens,
            "regions": page_regions,
        })

    return {
        "pages": pages_out,
        "num_pages": len(pages_out),
        "num_tokens": sum(len(p["tokens"]) for p in pages_out),
        "num_regions": sum(len(p["regions"]) for p in pages_out),
    }


def _build_result_zip(
    document: dict,
    md_text: str | None,
    extracted_images: list[dict],
) -> io.BytesIO:
    """Build an in-memory ZIP containing document.json, markdown.md, and images/."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # document.json
        zf.writestr("document.json", json.dumps(document, ensure_ascii=False, indent=2))

        # markdown.md (with relative image paths to images/)
        if md_text is not None:
            zf.writestr("markdown.md", md_text)

        # images/
        for img in extracted_images:
            zf.writestr(f"images/{img['filename']}", img["bytes"])

    buf.seek(0)
    return buf


# ── Routes ────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Check if the API and model are available."""
    return {
        "status": "ok",
        "model_available": DEFAULT_MODEL_PATH.exists(),
    }


@app.post("/parse")
async def parse_pdf(
    file: UploadFile = File(..., description="PDF file to parse"),
    include_markdown: bool = Query(True, description="Include markdown.md in the ZIP"),
    include_images: bool = Query(True, description="Include images/ folder in the ZIP"),
    model_path: str = Query("", description="Optional path to custom model file"),
    dpi: int = Query(150, ge=72, le=600, description="Render DPI for image processing"),
):
    """Parse a PDF and return a ZIP file.

    The ZIP contains:
      - document.json   — structured predictions with TSR nested under tables
      - markdown.md     — markdown with relative image references (images/*.png)
      - images/         — cropped region images as PNG files
    """
    # ── Save uploaded file to temp location ───────────────────────────
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    tmp_path = Path(tempfile.gettempdir()) / f"turboparse_{uuid.uuid4().hex}{suffix}"

    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        # ── Model path ────────────────────────────────────────────────
        mp = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        mp_str = str(mp) if mp.exists() else None

        # ── Run pipeline ──────────────────────────────────────────────
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

        # ── Build document.json ───────────────────────────────────────
        document = _build_document_json(
            pdf_features, predictions, all_figures, table_tsr_results,
        )

        # ── Markdown + images via converter ───────────────────────────
        md_text: str | None = None
        extracted_images: list[dict] = []

        if include_markdown or include_images:
            image_mode = "url" if include_images else "placeholder"
            converter = MarkdownConverter(
                pdf_features=pdf_features,
                predictions=predictions,
                regions=all_figures,
                table_tsr_results=table_tsr_results,
                pdf_path=str(tmp_path),
                pdf_document=pdf_doc,
                image_mode=image_mode,
            )

            if include_markdown:
                md_text = converter.convert()
            else:
                # Still need to trigger image extraction in url mode
                converter.convert()

            extracted_images = converter.extracted_images

        pdf_doc.close()

        # ── Build and return ZIP ──────────────────────────────────────
        zip_buf = _build_result_zip(document, md_text, extracted_images)

        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="result.zip"',
            },
        )

    except Exception as e:
        logger.exception("Parse failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "PDF parsing failed"},
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ── Run with: uvicorn api:app ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
