from pathlib import Path

import numpy as np
from tqdm import tqdm

from pdf_features.PdfToken import PdfToken
from pdf_token_type_labels.TokenType import TokenType
from .PdfTrainer import PdfTrainer
from .TokenFeatures import TokenFeatures, _real_tokens_cache, _sorted_by_top_cache
from .CRFDecoder import CRFDecoder

# Module-level cache: load CRF transition matrix once, reuse across PDFs
_crf_cache: dict[str, CRFDecoder] = {}


class TokenTypeTrainer(PdfTrainer):
    def get_model_input(self, memmap_path: str = None) -> np.ndarray:
        """Build the feature matrix.

        When memmap_path is None (default), uses an in-memory numpy array
        — faster for inference.  Pass an explicit path to use disk-backed
        memmap for large training sets.
        """
        contex_size = self.model_configuration.context_size

        # Clear stale cache entries from previous calls
        _real_tokens_cache.clear()
        _sorted_by_top_cache.clear()

        # Pre-build padding tokens once (reused across all pages)
        pad_left = [self.get_padding_token(segment_number=i - 999999, page_number=0) for i in range(contex_size)]
        pad_right = [self.get_padding_token(segment_number=999999 + i, page_number=0) for i in range(contex_size)]

        # ── Phase 1: count total rows (cheap, no feature computation) ──
        total_rows = 0
        for pdf_features in self.pdfs_features:
            for page in pdf_features.pages:
                if page.tokens:
                    total_rows += len(page.tokens)

        if total_rows == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # ── Phase 2: compute one row to determine feature dimension ──
        window_size = contex_size * 2
        sample_row = None
        for token_features, page in self.loop_token_features():
            if not page.tokens:
                continue
            for pt in pad_left:
                pt.page_number = page.page_number
            for pt in pad_right:
                pt.page_number = page.page_number
            page_tokens = pad_left + page.tokens + pad_right
            n_total = len(page_tokens)
            pair_features = [
                token_features.get_features(page_tokens[k], page_tokens[k + 1], page_tokens)
                for k in range(n_total - 1)
            ]
            i = contex_size
            start = i - contex_size
            sample_row = []
            for j in range(start, start + window_size):
                sample_row.extend(pair_features[j])
            break

        if sample_row is None:
            return np.zeros((0, 0), dtype=np.float32)

        n_features = len(sample_row)

        # ── Phase 3: build feature matrix ──
        # In-memory by default (fast for inference). Pass memmap_path to
        # use disk-backed storage for large training sets.
        if memmap_path is not None:
            print(f"Creating memmap: {total_rows:,} rows × {n_features} cols "
                  f"({total_rows * n_features * 4 / (1024**3):.1f} GB on disk)")
            X: np.ndarray = np.memmap(memmap_path, dtype=np.float32, mode='w+',
                          shape=(total_rows, n_features))
        else:
            X = np.empty((total_rows, n_features), dtype=np.float32)
        row_idx = 0

        # Clear caches again since loop_token_features creates new TokenFeatures
        _real_tokens_cache.clear()
        _sorted_by_top_cache.clear()

        for token_features, page in self.loop_token_features():
            if not page.tokens:
                continue

            for pt in pad_left:
                pt.page_number = page.page_number
            for pt in pad_right:
                pt.page_number = page.page_number

            page_tokens = pad_left + page.tokens + pad_right
            n_total = len(page_tokens)

            # ── Pre-compute per-token features (avoids double computation) ──
            # Each real token appears in 2 get_features() calls; these are
            # pure functions of the token alone — compute once and reuse.
            token_cache: dict[int, list[float]] = {}
            for token in page_tokens:
                tid = id(token)
                if tid not in token_cache:
                    token_cache[tid] = (
                        token_features.get_text_content_features(token)
                        + token_features.get_unicode_categories(token)
                        + token_features.get_image_overlap_features(token)
                    )

            # ── Build pair features into a flat numpy array ──────────
            # Using numpy avoids Python list-of-lists overhead.
            n_pairs = n_total - 1
            pair_array = np.empty((n_pairs, n_features // window_size), dtype=np.float32)
            for k in range(n_pairs):
                pair_array[k] = token_features.get_features_cached(
                    page_tokens[k], page_tokens[k + 1], page_tokens, token_cache
                )

            # ── Sliding window: numpy slicing instead of Python loops ──
            for i in range(contex_size, n_total - contex_size):
                start = i - contex_size
                X[row_idx] = pair_array[start:start + window_size].ravel()
                row_idx += 1

            # ── Prevent unbounded cache growth ──
            _real_tokens_cache.clear()
            _sorted_by_top_cache.clear()

        if memmap_path is not None:
            X.flush()
            print(f"Memmap written: {row_idx:,} rows to {memmap_path}")
        return X

    def loop_token_features(self):
        for pdf_features in tqdm(self.pdfs_features):
            token_features = TokenFeatures(pdf_features, token_image_overlaps=self.token_image_overlaps)

            for page in pdf_features.pages:
                if not page.tokens:
                    continue

                yield token_features, page

    def get_context_features(self, token_features: TokenFeatures, page_tokens: list[PdfToken], token_index: int):
        token_row_features = []
        first_token_from_context = token_index - self.model_configuration.context_size
        for i in range(self.model_configuration.context_size * 2):
            first_token = page_tokens[first_token_from_context + i]
            second_token = page_tokens[first_token_from_context + i + 1]
            token_row_features.extend(token_features.get_features(first_token, second_token, page_tokens))

        return token_row_features

    def predict(self, model_path: str | Path = None):
        predictions = super().predict(model_path)

        # ── P4.1: CRF decoding ─────────────────────────────────
        if self.model_configuration.use_crf:
            crf_path = str(model_path) + ".crf.npy" if model_path else None
            if crf_path and crf_path in _crf_cache:
                crf = _crf_cache[crf_path]
            elif crf_path and Path(crf_path).exists():
                crf = CRFDecoder.load(crf_path, self.model_configuration.num_class)
                _crf_cache[crf_path] = crf
            else:
                crf = CRFDecoder(self.model_configuration.num_class)

            predictions_assigned = 0
            for pdf_features in self.pdfs_features:
                for page in pdf_features.pages:
                    n = len(page.tokens)
                    if n == 0:
                        continue
                    page_probs = predictions[predictions_assigned : predictions_assigned + n]

                    # Viterbi decode this page's tokens (in reading order)
                    sorted_indices = sorted(range(n), key=lambda i: page.tokens[i].reading_order_no)
                    sorted_probs = page_probs[sorted_indices]
                    crf_labels = crf.decode(sorted_probs)
                    # Map back to original order
                    for orig_i, sorted_i in enumerate(sorted_indices):
                        page.tokens[sorted_i].prediction = crf_labels[orig_i]

                    predictions_assigned += n
        else:
            # Original argmax behavior
            predictions_assigned = 0
            for pdf_features in self.pdfs_features:
                for page in pdf_features.pages:
                    n = len(page.tokens)
                    if n == 0:
                        continue
                    for token, pred in zip(
                        page.tokens, predictions[predictions_assigned : predictions_assigned + n]
                    ):
                        token.prediction = int(np.argmax(pred))
                    predictions_assigned += n

    def set_token_types(self, model_path: str | Path = None):
        self.predict(model_path)
        for token in self.loop_tokens():
            token.token_type = TokenType.from_index(token.prediction)
