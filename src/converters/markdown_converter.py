"""
Convert PDF token predictions + TSR table structure → Markdown.

Usage:
    from markdown_converter import MarkdownConverter

    converter = MarkdownConverter(
        pdf_features=pdf_features,
        predictions=predictions,          # list[dict] from Token.to_dict()
        regions=all_figures,              # list[DetectedRegion]
        table_tsr_results=table_tsr_results,  # list[dict] with TSR XML
    )
    md_text = converter.convert()
    print(md_text)
"""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
import pypdfium2 as pdfium

from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from pdf_features.PdfPage import PdfPage
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from pdf_features.image_features.page_renderer import render_page_to_image


# ── Line: a horizontal row of tokens, or a region placeholder ────────
@dataclass
class Line:
    tokens: list[PdfToken] = field(default_factory=list)
    # For region placeholders (tables, figures) — set instead of tokens
    region_marker: str = ""          # "__REGION__Table", "__REGION__Picture", etc.
    region_bbox: Rectangle | None = None
    region_obj: object | None = None  # DetectedRegion reference

    @property
    def is_region(self) -> bool:
        return bool(self.region_marker)

    @property
    def top(self) -> int:
        if self.is_region and self.region_bbox:
            return self.region_bbox.top
        return min(t.bounding_box.top for t in self.tokens)

    @property
    def bottom(self) -> int:
        if self.is_region and self.region_bbox:
            return self.region_bbox.bottom
        return max(t.bounding_box.bottom for t in self.tokens)

    @property
    def left(self) -> int:
        if self.is_region and self.region_bbox:
            return self.region_bbox.left
        return min(t.bounding_box.left for t in self.tokens)

    @property
    def right(self) -> int:
        if self.is_region and self.region_bbox:
            return self.region_bbox.right
        return max(t.bounding_box.right for t in self.tokens)

    @property
    def text(self) -> str:
        return " ".join(t.content for t in self.tokens)

    @property
    def dominant_type(self) -> TokenType:
        """Most common token type on this line (cached)."""
        if self._dominant_type is None:
            if self.is_region:
                self._dominant_type = (
                    TokenType.TABLE if "Table" in self.region_marker
                    else TokenType.PICTURE
                )
            else:
                types = [TokenType.from_index(t.prediction) for t in self.tokens]
                self._dominant_type = Counter(types).most_common(1)[0][0]
        return self._dominant_type

    _dominant_type: TokenType | None = field(default=None, repr=False, init=False)


# ── Block: a group of consecutive lines forming a logical unit ────────
@dataclass
class Block:
    lines: list[Line] = field(default_factory=list)
    block_type: TokenType = TokenType.TEXT

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)


# ── Main converter ───────────────────────────────────────────────────
class MarkdownConverter:
    """Convert PDF token predictions and TSR results into Markdown."""

    # Vertical gap (in PDF points) above which we consider lines to
    # belong to different paragraphs / blocks.
    PARAGRAPH_GAP: int = 12

    # Minimum font-size ratio relative to the page's body font that
    # qualifies a line as a heading (even if the token type is TEXT).
    HEADING_FONT_RATIO: float = 1.3

    def __init__(
        self,
        pdf_features: PdfFeatures,
        predictions: list[dict] | None = None,
        regions: list | None = None,
        table_tsr_results: list[dict] | None = None,
        pdf_path: str | None = None,
        pdf_document: object | None = None,
    ):
        self.pdf_features = pdf_features
        self.predictions = predictions or []
        self.regions = regions or []
        self.table_tsr_results = table_tsr_results or []
        self.pdf_path = pdf_path

        # Reuse pre-opened document (avoids re-opening the same PDF)
        self._doc: object | None = pdf_document
        self._doc_is_external: bool = pdf_document is not None
        # Cache full-page renders: page_number → (RGB array, width, height, dpi)
        self._page_render_cache: dict[int, tuple] = {}

        # Pre-build token index by page — avoids re-scanning all pages
        # in _cells_to_markdown_table and _find_text_in_bbox.
        self._tokens_by_page: dict[int, list[tuple[int, str, int, int, int]]] = {}
        for page in self.pdf_features.pages:
            page_tokens: list[tuple[int, str, int, int, int]] = []
            for token in page.tokens:
                bb = token.bounding_box
                page_tokens.append((bb.left, token.content, bb.top, bb.right, bb.bottom))
            self._tokens_by_page[page.page_number] = page_tokens

        # Build lookup: (page, left, top) → token_type index
        self._prediction_map: dict[tuple[int, int, int], int] = {}
        for p in self.predictions:
            bb = p.get("bounding_box", {})
            key = (p["page_number"], bb.get("left", 0), bb.get("top", 0))
            # Store the index for the token type string
            try:
                tt = TokenType(p["token_type"])
                self._prediction_map[key] = tt.get_index()
            except (ValueError, KeyError):
                self._prediction_map[key] = TokenType.TEXT.get_index()

        # Build region index keyed by page
        self._regions_by_page: dict[int, list] = defaultdict(list)
        for r in self.regions:
            self._regions_by_page[r.page_number].append(r)

        # Build TSR index: (page, left, top) → tsr dict
        self._tsr_map: dict[tuple[int, int, int], dict] = {}
        for t in self.table_tsr_results:
            bb = t["bbox"]
            key = (t["page"], bb["left"], bb["top"])
            self._tsr_map[key] = t

        # Detect body font size (median font size of TEXT tokens)
        all_sizes = []
        for page in self.pdf_features.pages:
            for token in page.tokens:
                if (TokenType.from_index(token.prediction) == TokenType.TEXT
                        and token.font and token.font.font_size > 0):
                    all_sizes.append(token.font.font_size)
        self.body_font_size = (
            sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 10.0
        )

    # ── public API ────────────────────────────────────────────────────
    def convert(self) -> str:
        """Convert all pages to a single Markdown string."""
        parts: list[str] = []
        for page in self.pdf_features.pages:
            page_md = self._convert_page(page)
            if page_md.strip():
                parts.append(page_md)
        return "\n\n".join(parts)

    # ── per-page conversion ───────────────────────────────────────────
    def _convert_page(self, page: PdfPage) -> str:
        # Ensure predictions are applied to tokens
        self._apply_predictions(page)

        # Build lines
        lines = self._tokens_to_lines(page)

        # Insert region placeholders (tables / figures) at the right
        # vertical position among the lines
        lines = self._insert_region_blocks(page, lines)

        # Group lines into logical blocks
        blocks = self._lines_to_blocks(lines)

        # Render blocks to markdown
        return "\n\n".join(b for b in (self._block_to_md(b) for b in blocks) if b)

    # ── prediction application ────────────────────────────────────────
    def _apply_predictions(self, page: PdfPage) -> None:
        """Apply stored predictions to page tokens by best-match lookup."""
        pred_by_pos: dict[tuple[int, int], int] = {}
        for (pg, l, t), idx in self._prediction_map.items():
            if pg == page.page_number:
                pred_by_pos[(l, t)] = idx

        for token in page.tokens:
            key = (token.bounding_box.left, token.bounding_box.top)
            if key in pred_by_pos:
                token.prediction = pred_by_pos[key]
            else:
                # fuzzy: find closest prediction by left,top proximity
                best = None
                best_dist = float("inf")
                tl, tt = token.bounding_box.left, token.bounding_box.top
                for (l, t), idx in pred_by_pos.items():
                    dist = abs(l - tl) + abs(t - tt)
                    if dist < best_dist:
                        best_dist = dist
                        best = idx
                if best is not None and best_dist < 20:
                    token.prediction = best
                else:
                    token.prediction = TokenType.TEXT.get_index()

    # ── token → line grouping ─────────────────────────────────────────
    def _tokens_to_lines(self, page: PdfPage) -> list[Line]:
        """Group tokens into lines by vertical overlap, then sort."""
        remaining = list(page.tokens)
        raw_lines: list[list[PdfToken]] = []

        while remaining:
            pivot = remaining.pop(0)
            line_tokens = [pivot]
            # Collect all tokens on the same vertical band
            i = 0
            while i < len(remaining):
                if pivot.same_line(remaining[i]):
                    line_tokens.append(remaining.pop(i))
                else:
                    i += 1
            raw_lines.append(line_tokens)

        # Convert to Line objects, sort tokens left→right within each line
        lines: list[Line] = []
        for group in raw_lines:
            group.sort(key=lambda t: t.bounding_box.left)
            lines.append(Line(tokens=group))

        # Sort lines top→bottom
        lines.sort(key=lambda ln: ln.top)
        return lines

    # ── insert region blocks ──────────────────────────────────────────
    def _insert_region_blocks(
        self, page: PdfPage, lines: list[Line]
    ) -> list[Line]:
        """Insert pseudo-lines for detected Table / Picture regions."""
        page_regions = self._regions_by_page.get(page.page_number, [])
        if not page_regions:
            return lines

        result: list[Line] = []
        region_idx = 0
        page_regions.sort(key=lambda r: r.bbox.top)

        for line in lines:
            # Insert any regions whose top is above or at this line
            while (region_idx < len(page_regions)
                   and page_regions[region_idx].bbox.top <= line.bottom):
                region = page_regions[region_idx]
                region_idx += 1
                # Only insert blocks for tables and pictures
                if region.region_type in ("Table", "Picture", "Photo", "Figure", "Chart"):
                    result.append(self._region_to_pseudo_line(page, region))
            result.append(line)

        # Remaining regions after all lines
        while region_idx < len(page_regions):
            region = page_regions[region_idx]
            region_idx += 1
            if region.region_type in ("Table", "Picture", "Photo", "Figure", "Chart"):
                result.append(self._region_to_pseudo_line(page, region))

        return result

    def _region_to_pseudo_line(self, page: PdfPage, region) -> Line:
        """Create a special marker line for a detected region (no PdfToken needed)."""
        return Line(
            tokens=[],
            region_marker=f"__REGION__{region.region_type}__",
            region_bbox=region.bbox,
            region_obj=region,
        )

    # ── line → block grouping ─────────────────────────────────────────
    def _lines_to_blocks(self, lines: list[Line]) -> list[Block]:
        if not lines:
            return []

        blocks: list[Block] = []
        current = Block(lines=[], block_type=TokenType.TEXT)

        for i, line in enumerate(lines):
            # Region markers always form their own isolated block
            if line.is_region:
                if current.lines:
                    blocks.append(current)
                blocks.append(Block(lines=[line], block_type=line.dominant_type))
                current = Block(lines=[], block_type=TokenType.TEXT)
                continue

            # First text line starts a new block
            if not current.lines:
                current = Block(lines=[line], block_type=line.dominant_type)
                continue

            gap = line.top - current.lines[-1].bottom
            same_type = line.dominant_type == current.block_type

            # Heading-type lines always start a new block
            if line.dominant_type in (TokenType.TITLE, TokenType.SECTION_HEADER):
                blocks.append(current)
                current = Block(lines=[line], block_type=line.dominant_type)
                continue

            # Non-heading line after a heading block → always new block
            if current.block_type in (TokenType.TITLE, TokenType.SECTION_HEADER):
                blocks.append(current)
                current = Block(lines=[line], block_type=line.dominant_type)
                continue

            # Large gap → new block
            if gap > self.PARAGRAPH_GAP:
                blocks.append(current)
                current = Block(lines=[line], block_type=line.dominant_type)
                continue

            # Type change with gap
            if not same_type and gap > self.PARAGRAPH_GAP * 0.5:
                blocks.append(current)
                current = Block(lines=[line], block_type=line.dominant_type)
                continue

            current.lines.append(line)

        if current.lines:
            blocks.append(current)
        return blocks

    # ── block → markdown ──────────────────────────────────────────────
    def _block_to_md(self, block: Block) -> str:
        bt = block.block_type

        # ── Table region ──────────────────────────────────────────
        if bt == TokenType.TABLE:
            return self._render_table_block(block)

        # ── Picture / Figure region ───────────────────────────────
        if bt == TokenType.PICTURE:
            return self._render_picture_block(block)

        # ── Headings ──────────────────────────────────────────────
        if bt == TokenType.TITLE:
            return "# " + block.text
        if bt == TokenType.SECTION_HEADER:
            return "## " + block.text

        # ── List items ────────────────────────────────────────────
        if bt == TokenType.LIST_ITEM:
            return "\n".join(f"- {line.text}" for line in block.lines)

        # ── Caption ───────────────────────────────────────────────
        if bt == TokenType.CAPTION:
            return "*" + block.text + "*"

        # ── Formula ───────────────────────────────────────────────
        if bt == TokenType.FORMULA:
            return "$$\n" + block.text + "\n$$"

        # ── Footnote ──────────────────────────────────────────────
        if bt == TokenType.FOOTNOTE:
            return "[^note]: " + block.text

        # ── Page header / footer ──────────────────────────────────
        if bt in (TokenType.PAGE_HEADER, TokenType.PAGE_FOOTER):
            return f"<!-- {bt.value}: {block.text} -->"

        # ── Default: paragraph text ───────────────────────────────
        return block.text

    # ── table rendering ───────────────────────────────────────────────
    def _render_table_block(self, block: Block) -> str:
        """Render a table region as a Markdown table using TSR results."""
        region_line = block.lines[0] if block.lines else None
        if region_line is None:
            return ""

        region = region_line.region_obj
        if region is None:
            return ""

        # Look up TSR results for this region
        tsr_key = (region.page_number, region.bbox.left, region.bbox.top)
        tsr = self._tsr_map.get(tsr_key)
        if tsr is None:
            return f"<!-- Table detected but no TSR data -->\n\n"

        return self._tsr_xml_to_markdown(tsr, region)

    def _tsr_xml_to_markdown(self, tsr: dict, region) -> str:
        """Convert TSR XML to a Markdown table string using pre-computed cells."""
        # Prefer pre-computed cells (already in PDF points) over raw XML
        cells_list: list[dict] = tsr.get("cells", [])
        if cells_list:
            return self._cells_to_markdown_table(cells_list, region)

        # Fallback: parse XML (legacy path — cells will be in pixels, may be wrong)
        xml_str = tsr.get("xml", "")
        if not xml_str:
            return "<!-- Table: no XML -->\n\n"
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError:
            return "<!-- Table: invalid XML -->\n\n"

        cells_list = []
        for cell_el in root.findall("cell"):
            bb = cell_el.find("boundingbox")
            if bb is not None:
                cells_list.append({
                    "row": int(cell_el.get("row", 0)),
                    "col": int(cell_el.get("column", 0)),
                    "x": float(bb.get("x", 0)),
                    "y": float(bb.get("y", 0)),
                    "w": float(bb.get("w", 0)),
                    "h": float(bb.get("h", 0)),
                })
        return self._cells_to_markdown_table(cells_list, region)

    def _cells_to_markdown_table(self, cells_list: list[dict], region) -> str:
        """Build a markdown table from cell dicts (x/y/w/h in PDF points).

        Uses best-fit assignment: each token is assigned to the cell it
        overlaps the most — no padding, no duplication.
        """
        if not cells_list:
            return "<!-- Table: no cells -->\n\n"

        max_row = 0
        max_col = 0
        cell_boxes: list[tuple[int, int, int, int, int, int]] = []
        # (row, col, left, top, right, bottom) — all in PDF points

        for c in cells_list:
            row = int(c.get("row", 0))
            col = int(c.get("col", 0))
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            cell_boxes.append((
                row, col,
                region.bbox.left + c["x"],
                region.bbox.top + c["y"],
                region.bbox.left + c["x"] + c["w"],
                region.bbox.top + c["y"] + c["h"],
            ))

        # Use pre-built token index (O(1) lookup instead of O(pages) scan)
        page_tokens = self._tokens_by_page.get(region.page_number, [])

        # Proportional assignment: split tokens that span multiple cells
        cell_words: dict[tuple[int, int], list[tuple[int, str]]] = {
            (r, c): [] for r in range(max_row + 1) for c in range(max_col + 1)
        }

        for (tok_left, tok_text, tt, tr, tb) in page_tokens:
            tl = tok_left
            tok_area = (tr - tl) * (tb - tt)
            if tok_area <= 0:
                continue
            tok_width = tr - tl
            if tok_width <= 0:
                continue

            # Find all cells this token overlaps, with horizontal overlap
            overlaps: list[tuple[int, int, float]] = []
            for (row, col, cl, ct, cr, cb) in cell_boxes:
                ix1 = max(tl, cl)
                iy1 = max(tt, ct)
                ix2 = min(tr, cr)
                iy2 = min(tb, cb)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                if inter_area * 100.0 / tok_area < 10:
                    continue
                overlaps.append((row, col, float(ix2 - ix1)))

            if not overlaps:
                continue

            # Group by row, pick row with max total horizontal overlap
            row_map: dict[int, list[tuple[int, float]]] = {}
            for r, c, h in overlaps:
                row_map.setdefault(r, []).append((c, h))
            best_row = max(row_map, key=lambda r: sum(h for _, h in row_map[r]))
            row_cells = row_map[best_row]

            if len(row_cells) == 1:
                cell_words[(best_row, row_cells[0][0])].append((tok_left, tok_text))
            else:
                # Token spans multiple cells → split words proportionally
                total_h = sum(h for _, h in row_cells)
                words = tok_text.split()
                if not words:
                    continue
                row_cells.sort(key=lambda x: x[0])
                word_idx = 0
                for col, h_overlap in row_cells:
                    ratio = h_overlap / total_h
                    n = max(1, round(ratio * len(words)))
                    n = min(n, len(words) - word_idx)
                    if n <= 0:
                        continue
                    chunk = " ".join(words[word_idx:word_idx + n])
                    cell_words[(best_row, col)].append((tok_left, chunk))
                    word_idx += n
                    if word_idx >= len(words):
                        break

        # Build grid
        grid: dict[tuple[int, int], str] = {}
        for (r, c), words in cell_words.items():
            words.sort(key=lambda x: x[0])
            grid[(r, c)] = " ".join(w for _, w in words)

        # Render markdown table
        lines: list[str] = []
        for row in range(max_row + 1):
            row_cells = [grid.get((row, col), "") for col in range(max_col + 1)]
            lines.append("| " + " | ".join(row_cells) + " |")
            if row == 0:
                lines.append("| " + " | ".join("---" for _ in range(max_col + 1)) + " |")

        return "\n".join(lines)

    def _find_text_in_bbox(
        self, page_number: int, left: int, top: int, right: int, bottom: int
    ) -> str:
        """Find tokens overlapping the given box and join their text."""
        box_area = (right - left) * (bottom - top)
        if box_area <= 0:
            return ""
        # Expand cell bounds slightly to catch edge-clipped text
        pad = max(1, (right - left) // 20, (bottom - top) // 20)
        left -= pad
        top -= pad
        right += pad
        bottom += pad
        words: list[tuple[int, str]] = []
        for (tok_left, tok_text, tok_top, tok_right, tok_bottom) in self._tokens_by_page.get(page_number, []):
            ix1 = max(left, tok_left)
            iy1 = max(top, tok_top)
            ix2 = min(right, tok_right)
            iy2 = min(bottom, tok_bottom)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            token_area = (tok_right - tok_left) * (tok_bottom - tok_top)
            if token_area > 0 and (inter_area * 100 / token_area) > 10:
                words.append((tok_left, tok_text))
        words.sort(key=lambda x: x[0])
        return " ".join(w for _, w in words)

    # ── picture rendering ─────────────────────────────────────────────
    def _render_picture_block(self, block: Block) -> str:
        region_line = block.lines[0] if block.lines else None
        if region_line is None:
            return ""
        region = region_line.region_obj
        if region is None:
            return ""

        alt = f"{region.region_type} p{region.page_number}"
        bbox = region.bbox
        page_number = region.page_number

        # Try to embed the actual image as base64
        if self.pdf_path:
            try:
                if self._doc is None:
                    self._doc = pdfium.PdfDocument(self.pdf_path)
                doc = self._doc
                pi = page_number - 1
                if 0 <= pi < len(doc):
                    # Cache full-page renders — avoid re-rendering same page
                    if page_number not in self._page_render_cache:
                        dpi = 150
                        img = render_page_to_image(doc, pi, dpi=dpi)
                        h, w = img.shape[:2]
                        self._page_render_cache[page_number] = (img, w, h, dpi)

                    img, pix_w, pix_h, dpi = self._page_render_cache[page_number]
                    scale = dpi / 72.0
                    x1 = max(0, int(bbox.left * scale))
                    y1 = max(0, int(bbox.top * scale))
                    x2 = min(pix_w, int(bbox.right * scale))
                    y2 = min(pix_h, int(bbox.bottom * scale))

                    if x2 > x1 and y2 > y1:
                        crop = img[y1:y2, x1:x2]
                        _, buf = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                        b64 = base64.b64encode(buf).decode()
                        return f"![{alt}](data:image/png;base64,{b64})"
            except Exception:
                pass  # fall through to placeholder

        return (f"<!-- {region.region_type} at "
                f"({bbox.left},{bbox.top}) "
                f"{region.width}x{region.height} -->\n\n"
                f"![{alt}](image_page{region.page_number}_"
                f"{bbox.left}_{bbox.top}.png)")



# ── convenience function ──────────────────────────────────────────────
def predictions_to_markdown(
    pdf_features: PdfFeatures,
    predictions: list[dict],
    regions: list | None = None,
    table_tsr_results: list[dict] | None = None,
    pdf_path: str | None = None,
    pdf_document: object | None = None,
) -> str:
    """One-shot: convert predictions to Markdown.

    Pass pdf_document (a pypdfium2.PdfDocument) to reuse an already-open
    document instead of re-opening the PDF.
    """
    converter = MarkdownConverter(
        pdf_features=pdf_features,
        predictions=predictions,
        regions=regions or [],
        table_tsr_results=table_tsr_results or [],
        pdf_path=pdf_path,
        pdf_document=pdf_document,
    )
    return converter.convert()
