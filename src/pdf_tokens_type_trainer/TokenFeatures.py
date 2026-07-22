import bisect
import string
import unicodedata

from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from .config import CHARACTER_TYPE

# ── Pre-built lookup for O(1) unicode category → index ─────────────
_CHAR_TYPE_INDEX: dict[str, int] = {ct: i for i, ct in enumerate(CHARACTER_TYPE)}
_NUM_CHAR_TYPES = len(CHARACTER_TYPE)

# ── Module-level cache for real_tokens (list objects don't support __dict__) ──
_real_tokens_cache: dict[int, list] = {}

# ── Module-level cache: id(page_tokens) → tokens sorted by bounding_box.top ──
_sorted_by_top_cache: dict[int, list[PdfToken]] = {}


class TokenFeatures:
    def __init__(
        self,
        pdfs_features: PdfFeatures,
        token_image_overlaps: dict[tuple[int, float, float], bool] | None = None,
    ):
        self.pdfs_features = pdfs_features
        self.token_image_overlaps = token_image_overlaps or {}

        # Per-page cache for page-global features (computed once per page)
        self._page_cache: dict[int, dict] = {}

    # ── Main feature vector ──────────────────────────────────────────

    def get_features(self, token_1: PdfToken, token_2: PdfToken, page_tokens: list[PdfToken]):
        same_font = True if token_1.font.font_id == token_2.font.font_id else False

        return (
            # ── original base features (8) ──
            [
                same_font,
                self.pdfs_features.pdf_modes.font_size_mode / 100,
                len(token_1.content),
                len(token_2.content),
                token_1.content.count(" "),
                token_2.content.count(" "),
                sum(character in string.punctuation for character in token_1.content),
                sum(character in string.punctuation for character in token_2.content),
            ]
            # ── original position features (22) ──
            + self.get_position_features(token_1, token_2, page_tokens)
            # ── P2.2: text content features per token ──
            + self.get_text_content_features(token_1)
            + self.get_text_content_features(token_2)
            # ── P2.1: reading-order features ──
            + self.get_reading_order_features(token_1, token_2, page_tokens)
            # ── P2.3: page-global features (cached per page) ──
            + self.get_page_global_features(token_1, page_tokens)
            # ── P2.5: image overlap features ──
            + self.get_image_overlap_features(token_1)
            + self.get_image_overlap_features(token_2)
            # ── P2.4: expanded unicode categories (4+4 instead of 2+2) ──
            + self.get_unicode_categories(token_1)
            + self.get_unicode_categories(token_2)
        )

    def get_features_cached(
        self,
        token_1: PdfToken,
        token_2: PdfToken,
        page_tokens: list[PdfToken],
        token_cache: dict[int, list[float]],
    ) -> list[float]:
        """Fast path: per-token features (text_content, unicode, image_overlap)
        are pre-computed in token_cache — skip recomputing them.

        token_cache values are: text_content(12) + unicode(216) + image_overlap(1)

        Feature order MUST match get_features() exactly:
          base(8) + position(22) + text(t1,12) + text(t2,12) + reading(6)
          + page_global(7) + image(t1,1) + image(t2,1) + unicode(t1,216) + unicode(t2,216)
        """
        same_font = True if token_1.font.font_id == token_2.font.font_id else False

        # Per-pair features (depend on both tokens)
        base_pos = (
            [
                same_font,
                self.pdfs_features.pdf_modes.font_size_mode / 100,
                len(token_1.content),
                len(token_2.content),
                token_1.content.count(" "),
                token_2.content.count(" "),
                sum(character in string.punctuation for character in token_1.content),
                sum(character in string.punctuation for character in token_2.content),
            ]
            + self.get_position_features(token_1, token_2, page_tokens)
        )

        # Per-token features from cache: text(12) + unicode(216) + image(1)
        c1 = token_cache[id(token_1)]
        c2 = token_cache[id(token_2)]

        text_1, uni_1, img_1 = c1[:12], c1[12:228], c1[228:229]
        text_2, uni_2, img_2 = c2[:12], c2[12:228], c2[228:229]

        reading = self.get_reading_order_features(token_1, token_2, page_tokens)
        page_global = self.get_page_global_features(token_1, page_tokens)

        return base_pos + text_1 + text_2 + reading + page_global + img_1 + img_2 + uni_1 + uni_2

    # ── P2.2: Text content features ──────────────────────────────────

    @staticmethod
    def get_text_content_features(token: PdfToken) -> list[float]:
        """Rich text-content features beyond unicode categories."""
        content = token.content
        content_stripped = content.strip()
        font = token.font

        # Capitalisation signals
        is_all_caps = float(content_stripped.isupper() and len(content_stripped) >= 2)
        starts_with_capital = float(
            len(content_stripped) > 0 and content_stripped[0].isupper()
        )
        is_title_case = float(content_stripped.istitle() and len(content_stripped) >= 2)

        # Numeric signals
        is_numeric = float(content_stripped.replace(".", "").replace(",", "").isdigit() and len(content_stripped) > 0)
        contains_digit = float(any(ch.isdigit() for ch in content_stripped))
        digit_ratio = sum(ch.isdigit() for ch in content_stripped) / max(len(content_stripped), 1)

        # Punctuation endpoints
        ends_with_period = float(content_stripped.endswith("."))
        ends_with_colon = float(content_stripped.endswith(":"))
        ends_with_semicolon = float(content_stripped.endswith(";"))

        # Word count
        word_count = len(content_stripped.split()) if content_stripped else 0
        word_count_normalized = min(word_count / 20.0, 1.0)  # cap at 20 words

        # Font styling
        is_bold = float(font.bold)
        is_italic = float(font.italics)

        # Font size relative to page mode (how much bigger/smaller than normal)
        font_size = float(font.font_size)
        font_mode = float(font.font_size)  # token's own font size (direct access)
        font_size_ratio_to_page = font_size / max(font_size, 1.0)  # placeholder; real ratio below
        # Note: pdfs_features.pdf_modes.font_size_mode is the most-common font size on the page.
        # A token with font_size >> font_size_mode is likely a title or header.

        return [
            is_all_caps,
            starts_with_capital,
            is_title_case,
            is_numeric,
            contains_digit,
            digit_ratio,
            ends_with_period,
            ends_with_colon,
            ends_with_semicolon,
            word_count_normalized,
            is_bold,
            is_italic,
        ]

    # ── P2.1: Reading-order features ─────────────────────────────────

    @staticmethod
    def get_reading_order_features(
        token_1: PdfToken, token_2: PdfToken, page_tokens: list[PdfToken]
    ) -> list[float]:
        """Features based on reading order (logical sequence), not just spatial."""
        r1 = token_1.reading_order_no
        r2 = token_2.reading_order_no

        # Skip padding tokens
        if token_1.id == "pad_token" or token_2.id == "pad_token":
            return [0.0] * 6

        # Use module-level cache (list objects don't support __dict__)
        pt_id = id(page_tokens)
        real_tokens = _real_tokens_cache.get(pt_id)
        if real_tokens is None:
            real_tokens = [t for t in page_tokens if t.id != "pad_token"]
            _real_tokens_cache[pt_id] = real_tokens

        if not real_tokens:
            return [0.0] * 6

        ro_min = real_tokens[0].reading_order_no  # already sorted in get_model_input
        ro_max = real_tokens[-1].reading_order_no
        ro_range = max(ro_max - ro_min, 1)

        # Percentile position in reading order
        ro_percentile_1 = (r1 - ro_min) / ro_range
        ro_percentile_2 = (r2 - ro_min) / ro_range

        # Are tokens in consecutive reading order?
        are_ro_adjacent = float(abs(r2 - r1) <= 2)

        # Normalised gap
        ro_gap_norm = float(r2 - r1) / max(len(real_tokens), 1)

        return [
            ro_gap_norm,
            ro_percentile_1,
            ro_percentile_2,
            are_ro_adjacent,
            float(r1),
            float(r2),
        ]

    # ── P2.3: Page-global features (cached) ──────────────────────────

    def get_page_global_features(
        self, token: PdfToken, page_tokens: list[PdfToken]
    ) -> list[float]:
        """Features that describe the token's position in the global page layout."""
        page_id = id(page_tokens)

        if page_id not in self._page_cache:
            real_tokens = [t for t in page_tokens if t.id != "pad_token"]
            if not real_tokens:
                self._page_cache[page_id] = {
                    "page_token_count": 0.0,
                    "page_token_count_norm": 0.0,
                    "page_width": 1.0,
                    "page_height": 1.0,
                }
            else:
                # Estimate page dimensions from token bounding boxes
                page_right = max(t.bounding_box.right for t in real_tokens)
                page_bottom = max(t.bounding_box.bottom for t in real_tokens)
                page_left = min(t.bounding_box.left for t in real_tokens)
                page_top = min(t.bounding_box.top for t in real_tokens)

                self._page_cache[page_id] = {
                    "page_token_count": float(len(real_tokens)),
                    "page_token_count_norm": min(len(real_tokens) / 2000.0, 1.0),
                    "page_width": max(page_right - page_left, 1.0),
                    "page_height": max(page_bottom - page_top, 1.0),
                    "page_left": page_left,
                    "page_top": page_top,
                }

        cache = self._page_cache[page_id]

        # Horizontal position as fraction of page width
        x_frac = (token.bounding_box.left - cache["page_left"]) / cache["page_width"]
        x_frac_right = (token.bounding_box.right - cache["page_left"]) / cache["page_width"]

        # Vertical position as fraction of page height
        y_frac = (token.bounding_box.top - cache["page_top"]) / cache["page_height"]
        y_frac_bottom = (token.bounding_box.bottom - cache["page_top"]) / cache["page_height"]

        # Is near page top? (first ~10%)
        near_page_top = float(y_frac < 0.10)
        # Is near page bottom? (last ~10%)
        near_page_bottom = float(y_frac > 0.90)
        # Is near left edge?
        near_left_edge = float(x_frac < 0.05)
        # Is near right edge?
        near_right_edge = float(x_frac_right > 0.95)

        return [
            x_frac,
            y_frac,
            near_page_top,
            near_page_bottom,
            near_left_edge,
            near_right_edge,
            cache["page_token_count_norm"],
        ]

    # ── P2.5: Image overlap features ─────────────────────────────────

    def get_image_overlap_features(self, token: PdfToken) -> list[float]:
        """Binary feature: does this token overlap a detected image/table region?"""
        key = (token.page_number, token.bounding_box.left, token.bounding_box.top)
        overlaps_image = float(self.token_image_overlaps.get(key, False))
        return [overlaps_image]

    # ── Original position features (unchanged) ───────────────────────

    def get_position_features(self, token_1: PdfToken, token_2: PdfToken, page_tokens):
        left_1 = token_1.bounding_box.left
        right_1 = token_1.bounding_box.right
        height_1 = token_1.bounding_box.height
        width_1 = token_1.bounding_box.width

        left_2 = token_2.bounding_box.left
        right_2 = token_2.bounding_box.right
        height_2 = token_2.bounding_box.height
        width_2 = token_2.bounding_box.width

        right_gap_1, left_gap_2 = (
            token_1.pdf_token_context.left_of_token_on_the_right - right_1,
            left_2 - token_2.pdf_token_context.right_of_token_on_the_left,
        )

        absolute_right_1 = max(right_1, token_1.pdf_token_context.right_of_token_on_the_right)
        absolute_right_2 = max(right_2, token_2.pdf_token_context.right_of_token_on_the_right)

        absolute_left_1 = min(left_1, token_1.pdf_token_context.left_of_token_on_the_left)
        absolute_left_2 = min(left_2, token_2.pdf_token_context.left_of_token_on_the_left)

        right_distance, left_distance, height_difference = left_2 - left_1 - width_1, left_1 - left_2, height_1 - height_2

        top_distance = token_2.bounding_box.top - token_1.bounding_box.top - height_1
        top_distance_gaps = self.get_top_distance_gap(token_1, token_2, page_tokens)

        start_lines_differences = absolute_left_1 - absolute_left_2
        end_lines_difference = abs(absolute_right_1 - absolute_right_2)

        return [
            absolute_right_1,
            token_1.bounding_box.top,
            right_1,
            width_1,
            height_1,
            token_2.bounding_box.top,
            right_2,
            width_2,
            height_2,
            right_distance,
            left_distance,
            right_gap_1,
            left_gap_2,
            height_difference,
            top_distance,
            top_distance - self.pdfs_features.pdf_modes.lines_space_mode,
            top_distance_gaps,
            top_distance - height_1,
            end_lines_difference,
            start_lines_differences,
            self.pdfs_features.pdf_modes.lines_space_mode - top_distance_gaps,
            self.pdfs_features.pdf_modes.right_space_mode - absolute_right_1,
        ]

    @staticmethod
    def get_top_distance_gap(token_1: PdfToken, token_2: PdfToken, page_tokens):
        top_distance = token_2.bounding_box.top - token_1.bounding_box.top - token_1.bounding_box.height

        # Binary search: only scan tokens between t1.bottom and t2.top
        t1_bottom = token_1.bounding_box.bottom
        t2_top = token_2.bounding_box.top

        # Get or build sorted-by-top cache for this page_tokens list
        pt_id = id(page_tokens)
        sorted_tokens = _sorted_by_top_cache.get(pt_id)
        if sorted_tokens is None:
            sorted_tokens = sorted(
                [t for t in page_tokens if t.id != "pad_token"],
                key=lambda t: t.bounding_box.top,
            )
            _sorted_by_top_cache[pt_id] = sorted_tokens

        if not sorted_tokens:
            return top_distance

        # Find range: tokens with top in [t1_bottom, t2_top)
        tops = [t.bounding_box.top for t in sorted_tokens]
        lo = bisect.bisect_left(tops, t1_bottom)
        hi = bisect.bisect_left(tops, t2_top)

        if lo >= hi:
            return top_distance  # no middle tokens

        min_top = float("inf")
        max_bottom = float("-inf")
        for i in range(lo, hi):
            token = sorted_tokens[i]
            bt = token.bounding_box.top
            if bt < min_top:
                min_top = bt
            bb = token.bounding_box.bottom
            if bb > max_bottom:
                max_bottom = bb

        gap_middle_top = min_top - token_1.bounding_box.top - token_1.bounding_box.height
        gap_middle_bottom = t2_top - max_bottom
        return top_distance - (gap_middle_bottom - gap_middle_top)

    # ── P2.4: Expanded unicode categories (4+4 instead of 2+2) ───────

    @staticmethod
    def get_unicode_categories(token: PdfToken):
        # P2.4: use first 4 + last 4 characters (was 2+2)
        if token.id == "pad_token":
            return [-1] * _NUM_CHAR_TYPES * 8  # 8 slots × 27 categories

        content = token.content
        chars = content[:4] + content[-4:] if len(content) > 8 else content
        # Pad to exactly 8 characters
        chars = chars.ljust(8, '\0')[:8]

        # Pre-allocate result array (faster than list extending)
        result = [0] * (_NUM_CHAR_TYPES * 8)

        for slot, ch in enumerate(chars[:8]):
            if ch == '\0':
                continue
            cat = unicodedata.category(ch)
            idx = _CHAR_TYPE_INDEX.get(cat)
            if idx is not None:
                result[slot * _NUM_CHAR_TYPES + idx] = 1

        return result
