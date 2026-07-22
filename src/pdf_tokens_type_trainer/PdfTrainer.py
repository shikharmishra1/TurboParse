import os
from os.path import exists, join
from pathlib import Path

import lightgbm as lgb
import numpy as np

from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfFont import PdfFont
from pdf_features.PdfToken import PdfToken
from pdf_features.PdfTokenStyle import PdfTokenStyle
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from .ModelConfiguration import ModelConfiguration


# Module-level cache: load LightGBM model once, reuse across PDFs
_lgb_model_cache: dict[str, lgb.Booster] = {}


class PdfTrainer:
    def __init__(
        self,
        pdfs_features: list[PdfFeatures],
        model_configuration: ModelConfiguration = None,
        token_image_overlaps: dict[tuple[int, int, int], bool] | None = None,
    ):
        self.pdfs_features = pdfs_features
        self.model_configuration = model_configuration if model_configuration else ModelConfiguration()
        self.token_image_overlaps = token_image_overlaps or {}

    def get_model_input(self) -> np.ndarray:
        pass

    @staticmethod
    def features_rows_to_x(features_rows):
        if not features_rows:
            return np.zeros((0, 0))
        n_rows = len(features_rows)
        n_cols = len(features_rows[0])
        # float32 halves memory vs float64 (LightGBM handles float32 fine)
        x = np.zeros((n_rows, n_cols), dtype=np.float32)
        for i, row in enumerate(features_rows):
            x[i] = row
            features_rows[i] = None  # free the list row to keep peak memory low
        return x

    def train(self, model_path: str | Path, labels: list[int]):
        print(f"Getting model input")
        x_train = self.get_model_input()

        # Validate shapes before passing to LightGBM (prevents C-level crashes)
        n_samples, n_features = x_train.shape
        n_labels = len(labels)
        print(f"Features: {n_samples} rows × {n_features} cols, "
              f"labels: {n_labels}, dtype: {x_train.dtype}, "
              f"contiguous: {x_train.flags['C_CONTIGUOUS']}")

        if n_samples == 0 or n_features == 0:
            print("No data for training")
            return

        if n_samples != n_labels:
            raise ValueError(
                f"Mismatch: {n_samples} feature rows vs {n_labels} labels. "
                f"Re-run the labels cell to regenerate labels."
            )

        # Ensure C-contiguous float32 (required by LightGBM C API)
        if not x_train.flags['C_CONTIGUOUS']:
            x_train = np.ascontiguousarray(x_train)
        if x_train.dtype != np.float32:
            x_train = x_train.astype(np.float32)

        # ── P4.1: Build CRF transition matrix from training labels ──
        if self.model_configuration.use_crf:
            from .CRFDecoder import CRFDecoder

            # Group labels by page in reading order
            tokens_per_page = self._get_labels_per_page(labels)
            trans_matrix = CRFDecoder.build_transition_matrix(
                labels, tokens_per_page, self.model_configuration.num_class
            )
            crf = CRFDecoder(self.model_configuration.num_class, trans_matrix)
            crf_path = Path(str(model_path) + ".crf.npy")
            crf.save(str(crf_path))
            print(f"CRF transition matrix saved to {crf_path}")

        lgb_train = lgb.Dataset(x_train, labels)
        lgb_eval = lgb.Dataset(x_train, labels, reference=lgb_train)
        print(f"Training")

        if self.model_configuration.resume_training and exists(model_path):
            model = lgb.Booster(model_file=model_path)
            gbm = model.refit(x_train, labels)
        else:
            gbm = lgb.train(params=self.model_configuration.dict(), train_set=lgb_train, valid_sets=[lgb_eval])

        print(f"Saving")
        gbm.save_model(model_path, num_iteration=gbm.best_iteration)

    def loop_tokens(self):
        for pdf_features in self.pdfs_features:
            for page, token in pdf_features.loop_tokens():
                yield token

    @staticmethod
    def get_padding_token(segment_number: int, page_number: int):
        font = PdfFont(font_id="pad_font_id", bold=False, italics=False, font_size=0.0, color="#000000")
        return PdfToken(
            page_number=page_number,
            tag_id="pad_token",
            content="",
            pdf_font=font,
            reading_order_no=segment_number,
            bounding_box=Rectangle(left=0, top=0, right=0, bottom=0),
            token_type=TokenType.TEXT,
        )

    def predict(self, model_path: str | Path = None):
        model_path = str(model_path)
        x = self.get_model_input()

        if not x.any():
            return self.pdfs_features

        # Cache: load model from disk only once, reuse for subsequent PDFs
        if model_path not in _lgb_model_cache:
            _lgb_model_cache[model_path] = lgb.Booster(model_file=model_path)
        lightgbm_model = _lgb_model_cache[model_path]
        return lightgbm_model.predict(x)

    def save_training_data(self, save_folder_path: str | Path, labels: list[int]):
        os.makedirs(save_folder_path, exist_ok=True)

        x = self.get_model_input()

        np.save(join(str(save_folder_path), "x.npy"), x)
        np.save(join(str(save_folder_path), "y.npy"), labels)

    def _get_labels_per_page(self, labels: list[int]) -> list[list[int]]:
        """Group flat label list into per-page lists in reading order.

        Used by CRF to compute transition probabilities between adjacent
        tokens in reading order.
        """
        tokens_per_page: list[list[int]] = []
        idx = 0
        for pdf_features in self.pdfs_features:
            for page in pdf_features.pages:
                if not page.tokens:
                    continue
                # Sort tokens by reading order for correct adjacency
                sorted_tokens = sorted(page.tokens, key=lambda t: t.reading_order_no)
                n = len(sorted_tokens)
                page_labels = labels[idx : idx + n]
                tokens_per_page.append(page_labels)
                idx += n
        return tokens_per_page
