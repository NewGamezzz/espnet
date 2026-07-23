from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseCompressionModel, CompressionOutput


class DensityPeakCompression(BaseCompressionModel):
    """Frame boundary detection via Temporal-Aware Density Peak Clustering.

    Based on VARSTok (arXiv:2509.04685).  A frame is a good cluster seed if it
    is *locally dense* (surrounded by similar neighbours) **and** *isolated* from
    denser regions.  Concretely, each frame is scored as

        s[i] = rho[i] * delta[i]

    where

    * ``rho[i] = exp( mean of k-NN cosine similarities )``  - local density,
    * ``delta[i]`` - min distance to any frame j with rho[j] > rho[i];
      for the global density peak, ``delta = max distance to any frame``.

    Segmentation is the full VARSTok greedy bidirectional expansion: each
    iteration picks the highest-scored unassigned frame as a seed, then
    expands forward/backward up to *max_span* frames subject to a
    similarity threshold.  ``rate`` is used directly as that threshold:
    high rate (-> 1.0) makes merging hard (many segments, little
    compression); low rate (-> 0.0) makes merging easy (few segments,
    heavy compression).

    Args:
        k: Number of nearest neighbours for local density (default 5).
        beta: Penalty weight that discourages absorbing high-scored frames
              into a neighbouring cluster (default 0.2).
        max_span: Maximum frames per cluster (default 4).
    """

    def __init__(
        self,
        k: int = 5,
        beta: float = 0.2,
        max_span: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.beta = beta
        self.max_span = max_span

    # ------------------------------------------------------------------
    # Density-peak helpers (operate on a single, unpadded sequence)
    # ------------------------------------------------------------------

    def _similarity_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised pairwise cosine similarity in [0, 1].

        Args:
            x: (T, D)
        Returns:
            sim: (T, T)
        """
        x_norm = F.normalize(x, dim=-1)  # (T, D)
        cos = torch.matmul(x_norm, x_norm.T)  # (T, T)  in [-1, 1]
        return (cos + 1.0) / 2.0  # rescale to [0, 1]

    def _local_density(self, sim: torch.Tensor) -> torch.Tensor:
        """Local density rho = exp(mean of k-NN similarities).

        Args:
            sim: (T, T) similarity matrix
        Returns:
            rho: (T,)
        """
        T = sim.shape[0]
        k = min(self.k, T - 1)

        # Zero out self-similarity before selecting k nearest neighbours.
        sim_no_diag = sim.clone()
        sim_no_diag.fill_diagonal_(0.0)

        knn_sims, _ = torch.topk(sim_no_diag, k, dim=-1)  # (T, k)
        return torch.exp(knn_sims.mean(dim=-1))  # (T,)

    def _peak_distance(self, sim: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Peak distance delta.

        For frame i: delta[i] = min distance to any j with rho[j] > rho[i].
        For the global density peak (no such j): delta = max distance to any frame.

        Args:
            sim: (T, T) in [0, 1]
            rho: (T,)
        Returns:
            delta: (T,)
        """
        dist = 1.0 - sim  # (T, T)

        # (T, T)[i, j] = True when rho[j] > rho[i]
        higher = rho.unsqueeze(0) > rho.unsqueeze(1)  # (T, T)

        # Replace non-higher-density entries with inf so min ignores them.
        dist_masked = dist.masked_fill(~higher, float("inf"))
        delta, _ = dist_masked.min(dim=-1)  # (T,)

        # Frames with no denser neighbour get max distance (global peak rule).
        is_peak = delta.isinf()
        if is_peak.any():
            max_dist = dist.max(dim=-1).values  # (T,)
            delta = torch.where(is_peak, max_dist, delta)

        return delta

    def _density_scores(self, x: torch.Tensor):
        """Compute s = rho * delta for every frame.

        Args:
            x: (T, D) unpadded feature sequence
        Returns:
            s:   (T,)   clustering score
            sim: (T, T) similarity matrix (kept for greedy clustering)
            rho: (T,)   local density (kept for greedy clustering)
        """
        sim = self._similarity_matrix(x)
        rho = self._local_density(sim)
        delta = self._peak_distance(sim, rho)
        return rho * delta, sim, rho

    # ------------------------------------------------------------------
    # Greedy bidirectional clustering
    # ------------------------------------------------------------------

    def _greedy_cluster(
        self,
        s: torch.Tensor,
        sim: torch.Tensor,
        max_span: int,
        threshold: float,
    ) -> torch.Tensor:
        """Greedy bidirectional expansion from density peaks.

        Iteratively selects the highest-scored unassigned frame as a cluster
        seed and expands forwards and backwards up to *max_span* frames,
        merging any frame whose adjusted similarity exceeds *threshold*.

        The adjusted similarity penalises absorbing high-scored frames::

            sim_score[seed, t] = sim[seed, t] - beta * s[t]

        Args:
            s:        (T,) density scores
            sim:      (T, T) similarity matrix in [0, 1]
            max_span: maximum frames per cluster
            threshold: similarity cutoff for merging
        Returns:
            segment_idx: (T,) integer segment index, temporally ordered
        """
        T = s.shape[0]
        assigned = torch.zeros(T, dtype=torch.bool, device=s.device)
        clusters: list[list[int]] = []

        scores = s.clone()

        while not assigned.all():
            # Pick the unassigned frame with the highest density score.
            masked = scores.clone()
            masked[assigned] = -float("inf")
            seed = int(masked.argmax().item())

            cluster = [seed]
            assigned[seed] = True

            # Penalise adding other high-score frames into this cluster.
            sim_score = sim[seed] - self.beta * s  # (T,)

            # Forward expansion.
            for t in range(seed + 1, min(T, seed + max_span + 1)):
                if assigned[t] or len(cluster) >= max_span:
                    break
                if sim_score[t].item() > threshold:
                    cluster.append(t)
                    assigned[t] = True
                else:
                    break

            # Backward expansion.
            for t in range(seed - 1, max(-1, seed - max_span - 1), -1):
                if assigned[t] or len(cluster) >= max_span:
                    break
                if sim_score[t].item() > threshold:
                    cluster.append(t)
                    assigned[t] = True
                else:
                    break

            clusters.append(sorted(cluster))

        # Sort clusters by their earliest frame to restore temporal order.
        clusters.sort(key=lambda c: c[0])

        segment_idx = torch.zeros(T, dtype=torch.long, device=s.device)
        for seg_id, cluster in enumerate(clusters):
            for t in cluster:
                segment_idx[t] = seg_id

        return segment_idx

    # ------------------------------------------------------------------
    # Shared helpers (mirrored from CosineSimilarityCompression)
    # ------------------------------------------------------------------

    def _no_rate_output(self, x: torch.Tensor, padding_mask) -> CompressionOutput:
        """Trivial per-frame segmentation returned when rate is None."""
        B, T, _ = x.shape
        device = x.device
        boundary = torch.ones(B, T, device=device, dtype=torch.long)
        boundary[:, 0] = 0
        if padding_mask is not None:
            boundary = boundary.masked_fill(padding_mask, 0)
        boundary_soft = boundary.float()
        segment_idx = torch.cumsum(boundary, dim=1)
        expected_length = boundary_soft.sum(dim=1, keepdim=True) + 1
        return CompressionOutput(
            boundary_soft=boundary_soft,
            segment_idx=segment_idx,
            reconstructed_features=x,
            expected_length=expected_length,
        )

    def _segment_average(
        self,
        x: torch.Tensor,
        segment_idx: torch.Tensor,
        padding_mask,
    ) -> torch.Tensor:
        """Segment-wise mean features, upsampled back to frame resolution.

        Returns:
            (B, T, D)
        """
        device = x.device
        max_segments = int(segment_idx.max().item()) + 1
        seg_ids = torch.arange(max_segments, device=device)[None, :, None]  # (1, S, 1)
        assign = (segment_idx.unsqueeze(1) == seg_ids).float()  # (B, S, T)
        if padding_mask is not None:
            assign = assign.masked_fill(padding_mask.unsqueeze(1), 0.0)
        assign_norm = assign / (assign.sum(dim=2, keepdim=True) + 1e-6)
        segment_means = torch.matmul(assign_norm, x)  # (B, S, D)
        return torch.matmul(assign.transpose(1, 2), segment_means)  # (B, T, D)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        rate=None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressionOutput:
        """Segment the input sequence using density peak clustering.

        Args:
            x:            (B, T, D) input features.
            rate:         Compression rate in (0, 1], used directly as the
                          similarity threshold for cluster merging;
                          ``max_span`` is fixed at construction time.  Pass
                          ``None`` to return a trivial per-frame identity
                          segmentation.
            padding_mask: (B, T) bool tensor; True marks padding positions.

        Returns:
            CompressionOutput
        """
        B, T, _ = x.shape
        device = x.device

        if rate is None:
            return self._no_rate_output(x, padding_mask)

        valid_len = (
            (~padding_mask).sum(dim=1).float() if padding_mask is not None else None
        )

        if isinstance(rate, torch.Tensor):
            rate_b = rate.squeeze(-1)  # (B,) or scalar tensor
        else:
            rate_b = torch.full((B,), rate, device=device, dtype=torch.float)

        all_segment_idx = torch.zeros(B, T, dtype=torch.long, device=device)

        for b in range(B):
            vlen = int(valid_len[b].item()) if valid_len is not None else T
            if vlen <= 1:
                continue

            x_b = x[b, :vlen]  # (vlen, D)

            s_b, sim_b, _ = self._density_scores(x_b)

            # rate is used directly as the similarity threshold:
            #   high rate (-> 1.0) = hard to merge = many segments
            #   low  rate (-> 0.0) = easy to merge = few  segments
            threshold = float(rate_b[b].item() if rate_b.dim() > 0 else rate_b.item())

            seg_idx_b = self._greedy_cluster(s_b, sim_b, self.max_span, threshold)
            all_segment_idx[b, :vlen] = seg_idx_b

            # Padding frames share the last valid segment.
            if vlen < T:
                all_segment_idx[b, vlen:] = int(seg_idx_b[-1].item())

        # Derive boundary tensor from segment indices.
        boundary = torch.zeros(B, T, device=device, dtype=torch.long)
        boundary[:, 1:] = (all_segment_idx[:, 1:] != all_segment_idx[:, :-1]).long()
        if padding_mask is not None:
            boundary = boundary.masked_fill(padding_mask, 0)

        boundary_soft = boundary.float()
        # Recompute cumsum-based segment_idx for consistency with the rest of
        # the pipeline (same convention as CosineSimilarityCompression).
        segment_idx = torch.cumsum(boundary, dim=1)
        reconstructed = self._segment_average(x, segment_idx, padding_mask)
        expected_length = boundary_soft.sum(dim=1, keepdim=True) + 1

        return CompressionOutput(
            boundary_soft=boundary_soft,
            segment_idx=segment_idx,
            reconstructed_features=reconstructed,
            expected_length=expected_length,
        )
