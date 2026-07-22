"""CRF (Conditional Random Field) decoder for sequence-consistent token type prediction.

After LightGBM outputs per-token probability distributions, this module applies
Viterbi decoding with learned transition probabilities to enforce label consistency
across the reading-order sequence.

Transition probabilities are learned from training data: how often does label B
follow label A for adjacent tokens in reading order?
"""

import numpy as np


class CRFDecoder:
    """Viterbi decoder with emission + transition potentials."""

    def __init__(self, num_classes: int, transition_matrix: np.ndarray | None = None):
        """
        Args:
            num_classes: Number of token type classes.
            transition_matrix: (num_classes, num_classes) transition counts or
                probabilities. If None, defaults to uniform.
        """
        self.num_classes = num_classes
        if transition_matrix is not None:
            self.transition_matrix = transition_matrix.astype(np.float64)
        else:
            self.transition_matrix = np.ones((num_classes, num_classes), dtype=np.float64)

        # Normalize rows → transition probabilities (add-one smoothing)
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_probs = self.transition_matrix / np.maximum(row_sums, 1)

        # Precompute log probabilities (add small epsilon to avoid log(0))
        eps = 1e-12
        self.log_transition = np.log(self.transition_probs + eps)

    @staticmethod
    def build_transition_matrix(
        labels: list[int], tokens_per_page: list[list[int]], num_classes: int
    ) -> np.ndarray:
        """Build transition count matrix from training labels.

        Args:
            labels: Flat list of all training labels in reading order.
            tokens_per_page: List where each element is a list of label indices
                for one page, in reading order.
            num_classes: Total number of token type classes.

        Returns:
            (num_classes, num_classes) transition count matrix.
        """
        trans = np.ones((num_classes, num_classes), dtype=np.float64)  # add-one smoothing

        for page_labels in tokens_per_page:
            for i in range(len(page_labels) - 1):
                prev = page_labels[i]
                curr = page_labels[i + 1]
                if 0 <= prev < num_classes and 0 <= curr < num_classes:
                    trans[prev, curr] += 1

        return trans

    def decode(self, emissions: np.ndarray) -> list[int]:
        """Viterbi decoding of the most likely label sequence.

        Args:
            emissions: (sequence_length, num_classes) matrix of emission
                probabilities from LightGBM. Each row should sum to ~1.

        Returns:
            List of predicted class indices (length = sequence_length).
        """
        n = emissions.shape[0]
        if n == 0:
            return []
        if n == 1:
            return [int(np.argmax(emissions[0]))]

        # Log-space to avoid underflow
        eps = 1e-12
        log_emissions = np.log(emissions + eps)  # (n, num_classes)

        # Viterbi DP
        # dp[t][j] = max score ending at position t with label j
        # backptr[t][j] = previous label that gave the max
        dp = np.zeros((n, self.num_classes), dtype=np.float64)
        backptr = np.zeros((n, self.num_classes), dtype=np.int32)

        # Initialize first position
        dp[0] = log_emissions[0]

        for t in range(1, n):
            for j in range(self.num_classes):
                # Score for each possible previous label k:
                # dp[t-1][k] + log_transition[k][j] + log_emissions[t][j]
                scores = dp[t - 1] + self.log_transition[:, j] + log_emissions[t, j]
                best_k = int(np.argmax(scores))
                dp[t, j] = scores[best_k]
                backptr[t, j] = best_k

        # Backtrack
        best_last = int(np.argmax(dp[-1]))
        path = [best_last]
        for t in range(n - 1, 0, -1):
            best_last = backptr[t, best_last]
            path.append(best_last)

        return list(reversed(path))

    def save(self, filepath: str) -> None:
        """Save the transition matrix to a .npy file."""
        np.save(filepath, self.transition_matrix)

    @classmethod
    def load(cls, filepath: str, num_classes: int) -> "CRFDecoder":
        """Load a transition matrix from a .npy file."""
        trans = np.load(filepath)
        return cls(num_classes=num_classes, transition_matrix=trans)
