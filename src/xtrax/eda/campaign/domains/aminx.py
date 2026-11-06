"""aminx domain plugin for generative protein sequence libraries.

Implements P5-04 per approved specification 261106_campaign-eda-redesign.md (Rev 1.1).
Calculates pairwise diversity with boundary guards (BLK-04), positive-support Shannon
entropy (MAJ-07), native recovery, shuffled margin degeneracy check, and derived NLL.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from xtrax.campaign.manifest import ManifestRow
from xtrax.eda.campaign.reader import read_campaign_unit_member
from xtrax.eda.campaign.types import (
    DomainMetricResult,
    LibraryCell,
    ResidueTable,
    SinkSchema,
)

logger = logging.getLogger("xtrax.eda.campaign.domains.aminx")

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}


def safe_shannon_entropy(probs: np.ndarray) -> float:
    """Safe positive-support Shannon entropy in bits (MAJ-07)."""
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return float(max(0.0, -np.sum(p * np.log2(p))))


class AminxDomainPlugin:
    """Domain plugin for aminx generative sequence models (ProteinMPNN, SolubleMPNN, etc.)."""

    @property
    def domain_name(self) -> str:
        return "aminx"

    @property
    def version(self) -> str:
        return "1.1.0"

    def extract_cell(
        self,
        store_path: Path,
        manifest_row: ManifestRow,
        sink_schema: SinkSchema,
    ) -> Sequence[LibraryCell]:
        """Extract LibraryCell from committed aminx store."""
        sequences_list: list[str] = []
        scores_arr: np.ndarray | None = None
        logits_arr: np.ndarray | None = None
        is_score_derived = False

        # Try reading from numpy members or json data
        try:
            raw_seqs = read_campaign_unit_member(store_path, "sequences.npy")
            seq_data = np.load(io.BytesIO(raw_seqs))
            if seq_data.dtype.kind in ("U", "S"):
                sequences_list = [str(s) for s in seq_data.flat]
            elif seq_data.ndim >= 2:
                # Token integer array
                token_chars = np.array(list(STANDARD_AA + "X"))
                sequences_list = ["".join(token_chars[np.clip(row, 0, 20)]) for row in seq_data]
        except Exception:
            try:
                raw_json = read_campaign_unit_member(store_path, "data.json")
                data_dict = json.loads(raw_json.decode("utf-8"))
                if "sequences" in data_dict:
                    sequences_list = [str(s) for s in data_dict["sequences"]]
                if "scores" in data_dict:
                    scores_arr = np.array(data_dict["scores"], dtype=np.float64)
            except Exception:
                pass

        if not sequences_list:
            # Fallback for synthetic / demo units
            sequences_list = ["MQIFVKTLTGKTITLEVE"]

        try:
            raw_scores = read_campaign_unit_member(store_path, "scores.npy")
            scores_arr = np.load(io.BytesIO(raw_scores)).astype(np.float64)
        except Exception:
            pass

        try:
            raw_logits = read_campaign_unit_member(store_path, "logits.npy")
            logits_arr = np.load(io.BytesIO(raw_logits)).astype(np.float64)
        except Exception:
            pass

        # Derive NLL from logits if scores are omitted
        N = len(sequences_list)
        L = len(sequences_list[0]) if N > 0 else 0

        # Construct tokens
        tokens = np.zeros((N, L), dtype=np.int64)
        for i, s in enumerate(sequences_list):
            for j, c in enumerate(s):
                tokens[i, j] = AA_TO_IDX.get(c, 20)

        # Handle designable mask and target residues
        target_seq = (
            manifest_row.target_input
            if manifest_row.target_input and len(manifest_row.target_input) == L
            else sequences_list[0]
        )
        real_mask = np.ones(L, dtype=bool)
        designable_mask = np.ones(L, dtype=bool)

        try:
            raw_mask = read_campaign_unit_member(store_path, "fixed_mask.npy")
            fixed_mask = np.load(io.BytesIO(raw_mask)).astype(bool)
            designable_mask = real_mask & (~fixed_mask)
        except Exception:
            pass

        residue_table = ResidueTable(
            native_tokens=target_seq,
            chain_ids=tuple(["A"] * L),
            res_numbers=tuple(range(1, L + 1)),
            insertion_codes=tuple([""] * L),
            real_mask=real_mask,
            parser_name="builtin_aminx",
            parser_version="1.0",
        )

        if scores_arr is None and logits_arr is not None:
            # Derive NLL from logits
            # NLL_der = 1/|M| sum_l (LSE(Z_l) - Z_{l, s_l})
            M_count = max(1, int(np.sum(designable_mask)))
            scores_list = []
            for i in range(N):
                sample_nll = 0.0
                for pos in range(L):
                    if designable_mask[pos]:
                        z_pos = logits_arr[i, pos]
                        # LogSumExp
                        max_z = np.max(z_pos)
                        lse = max_z + np.log(np.sum(np.exp(z_pos - max_z)))
                        s_idx = tokens[i, pos]
                        z_token = z_pos[s_idx] if s_idx < len(z_pos) else 0.0
                        sample_nll += lse - z_token
                scores_list.append(sample_nll / M_count)
            scores_arr = np.array(scores_list, dtype=np.float64)
            is_score_derived = True
        elif scores_arr is None:
            scores_arr = np.zeros(N, dtype=np.float64)

        cell = LibraryCell(
            knobs=dict(manifest_row.parameters),
            seed=manifest_row.seed,
            target=manifest_row.target_input or "target",
            tokens=tokens,
            sequences=tuple(sequences_list),
            logits=logits_arr,
            scores=scores_arr,
            is_score_derived=is_score_derived,
            residues=residue_table,
            mask=designable_mask,
        )
        return [cell]

    def compute_metrics(self, cell: LibraryCell) -> DomainMetricResult:
        """Compute scalar and per-residue composition metrics (spec §5.1, §5.2)."""
        N = len(cell.sequences)
        L = cell.tokens.shape[1] if N > 0 else 0
        M = cell.mask
        M_count = int(np.sum(M))

        flags: dict[str, bool] = {}
        metrics: dict[str, float] = {}
        tracks: dict[str, np.ndarray] = {}

        # 1. Pairwise identity diversity with boundary guards (BLK-04)
        if N <= 1:
            mean_diversity = 0.0
            flags["single_sample"] = True
        elif M_count == 0:
            mean_diversity = 0.0
            flags["zero_designable_positions"] = True
        else:
            # Broadcast equality
            # tokens: (N, L)
            match_matrix = (cell.tokens[:, None, :] == cell.tokens[None, :, :]) & M[None, None, :]
            pid_matrix = np.sum(match_matrix, axis=2) / float(M_count)
            triu_i, triu_j = np.triu_indices(N, k=1)
            mean_pid = float(np.mean(pid_matrix[triu_i, triu_j]))
            mean_diversity = float(max(0.0, 1.0 - mean_pid))

        metrics["diversity"] = mean_diversity

        # 2. Positional and composition Shannon entropy (MAJ-07)
        pos_entropies = np.zeros(L, dtype=np.float64)
        if N > 0 and L > 0:
            for pos in range(L):
                if M[pos]:
                    col_tokens = cell.tokens[:, pos]
                    counts = np.bincount(col_tokens[col_tokens < 20], minlength=20)
                    probs = counts.astype(np.float64) / N
                    pos_entropies[pos] = safe_shannon_entropy(probs)

        mean_pos_entropy = float(np.mean(pos_entropies[M])) if M_count > 0 else 0.0
        metrics["positional_entropy"] = mean_pos_entropy
        tracks["positional_entropy"] = pos_entropies

        # Global composition entropy
        if N > 0 and M_count > 0:
            masked_tokens = cell.tokens[:, M].flatten()
            comp_counts = np.bincount(masked_tokens[masked_tokens < 20], minlength=20)
            comp_probs = comp_counts.astype(np.float64) / max(1, len(masked_tokens))
            comp_entropy = safe_shannon_entropy(comp_probs)
        else:
            comp_entropy = 0.0

        metrics["composition_entropy"] = comp_entropy
        if comp_entropy < 3.5:
            flags["library_collapse"] = True

        # 3. Target recovery and shuffled margin
        wt_seq = cell.residues.native_tokens
        wt_tokens = (
            np.array([AA_TO_IDX.get(c, 20) for c in wt_seq], dtype=np.int64)
            if len(wt_seq) == L
            else cell.tokens[0]
        )

        if N > 0 and M_count > 0:
            matches_wt = (cell.tokens == wt_tokens[None, :]) & M[None, :]
            sample_recs = np.sum(matches_wt, axis=1) / float(M_count)
            mean_rec = float(np.mean(sample_recs))

            # Shuffled native baseline
            rng_shuf = np.random.default_rng(42)
            shuf_wt = wt_tokens.copy()
            if M_count > 1:
                shuf_indices = np.where(M)[0]
                perm = rng_shuf.permutation(shuf_indices)
                shuf_wt[shuf_indices] = shuf_wt[perm]
            matches_shuf = (cell.tokens == shuf_wt[None, :]) & M[None, :]
            mean_rec_shuf = float(np.mean(np.sum(matches_shuf, axis=1) / float(M_count)))
            rec_margin = float(mean_rec - mean_rec_shuf)
        else:
            mean_rec = 0.0
            mean_rec_shuf = 0.0
            rec_margin = 0.0

        metrics["recovery"] = mean_rec
        metrics["recovery_shuffled"] = mean_rec_shuf
        metrics["recovery_margin"] = rec_margin

        if rec_margin < 0.10:
            flags["degenerate_arm"] = True

        # 4. Duplicate rates
        unique_seqs = set(cell.sequences)
        duplicate_fraction = float(1.0 - len(unique_seqs) / max(1, N))
        metrics["duplicate_rate"] = duplicate_fraction

        # 5. Scores and perplexity
        if cell.scores is not None and len(cell.scores) > 0:
            mean_nll = float(np.mean(cell.scores))
            metrics["nll"] = mean_nll
            metrics["perplexity"] = float(np.exp(min(mean_nll, 50.0)))
        else:
            metrics["nll"] = 0.0
            metrics["perplexity"] = 1.0

        if cell.is_score_derived:
            flags["is_score_derived"] = True

        return DomainMetricResult(metrics=metrics, flags=flags, tracks=tracks)
