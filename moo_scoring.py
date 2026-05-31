#!/usr/bin/env python3
"""
Weighted Multi-Objective Scoring for Layout Optimization

This module provides a unified scoring system that combines direct 
position-pair scoring table lookup with item-pair score weighting, 
specifically designed for multi-objective optimization of layouts.

Core Features:
- Supports arbitrary number of objectives in a position-pair scoring table
- Direct score lookup with weighting
- Optimized for partial layout scoring during search

Usage:
    scorer = WeightedMOOScorer(
        objectives=['engram_rows', 'engram_keys'],
        position_pair_score_table='input/engram_2key_scores.csv',
        items=['e', 't', 'a', 'o'],
        positions=['F', 'D', 'S', 'J']
    )
    
    scores = scorer.score_layout(mapping_array)  # Returns [obj1_score, obj2_score, ...]
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from combos import parse_slot, count_same_finger_pairs, count_cross_same_finger_pairs


@dataclass
class ScoringArrays:
    """Minimal compatibility wrapper for existing search infrastructure."""
    item_scores: np.ndarray
    item_pair_matrix: np.ndarray  
    position_matrix: np.ndarray
    
    def __post_init__(self):
        self.n_items = len(self.item_scores)
        self.n_positions = self.position_matrix.shape[0]


class WeightedMOOScorer:
    """
    Multi-objective scorer supporting both bigram and trigram objectives.
    """
    
    def __init__(self, objectives: List[str], position_pair_score_table: str,
                 items: List[str], positions: List[str], 
                 weights: Optional[List[float]] = None, 
                 maximize: Optional[List[bool]] = None,
                 item_pair_score_table: str = "input/normalized-english-letter-pair-counts-google-ngrams.csv",
                 position_triple_score_table: Optional[str] = None,
                 item_triple_score_table: Optional[str] = None,
                 combo_penalty: float = 0.5,
                 max_combo_size: int = 2,
                 combo_same_finger_penalty: float = 0.5,
                 verbose: bool = False):
        """
        Initialize weighted MOO scorer with bigram and trigram support.
        
        Args:
            objectives: List of objective names
            position_pair_score_table: Path to CSV with bigram position scores
            items: List of items being optimized
            positions: List of available positions  
            weights: Optional weights for each objective
            maximize: Optional direction for each objective
            item_pair_score_table: Path to English bigram frequencies
            position_triple_score_table: Path to CSV with trigram position scores
            item_triple_score_table: Path to English trigram frequencies
            combo_penalty: Multiplier applied as combo_penalty^(slot_size - 2) when scoring slots whose size > 2.
            max_combo_size: Maximum number of constituent keys for any combo slot.
        """
        self.objectives = objectives
        self.items = [item.upper() for item in items]
        # Positions may include combo slot IDs (bracketed); preserve them as-is.
        self.positions = [pos.upper() if not (pos.startswith('[') and pos.endswith(']')) else pos.upper() for pos in positions]
        self.objective_weights = weights or [1.0] * len(objectives)
        self.objective_maximize = maximize or [True] * len(objectives)
        self.combo_penalty = combo_penalty
        self.max_combo_size = max_combo_size
        self.combo_same_finger_penalty = combo_same_finger_penalty

        # Pre-parse each slot into its constituent single-key chars, e.g. 'F' -> ('F',), '[DF]' -> ('D','F')
        self.position_constituents: List[Tuple[str, ...]] = [
            tuple(c.upper() for c in parse_slot(p)) for p in self.positions
        ]
        # Pre-compute the number of same-finger constituent pairs per slot (0 for single keys)
        self.position_same_finger_pairs: List[int] = [
            count_same_finger_pairs(c) for c in self.position_constituents
        ]
        # Pre-compute cross-slot same-finger constituent pair counts. Used to
        # penalise "combo SFBs": a bigram transition between slot i and slot j
        # where some constituent of i and some (different) constituent of j
        # share a finger. Only applied when at least one of the two slots is a
        # combo, otherwise single-key SFBs would be double-counted (they are
        # already encoded in the engram_same_finger table).
        n_pos_init = len(self.positions)
        self._cross_sf_pairs = np.zeros((n_pos_init, n_pos_init), dtype=np.int32)
        for i in range(n_pos_init):
            ci = self.position_constituents[i]
            for j in range(n_pos_init):
                if i == j:
                    continue
                # Skip single-vs-single: the position-pair table already handles it.
                if len(ci) == 1 and len(self.position_constituents[j]) == 1:
                    continue
                self._cross_sf_pairs[i, j] = count_cross_same_finger_pairs(
                    ci, self.position_constituents[j]
                )
        
        # Validate inputs
        if len(self.objective_weights) != len(objectives):
            raise ValueError(f"Weights length ({len(self.objective_weights)}) != objectives length ({len(objectives)})")
        if len(self.objective_maximize) != len(objectives):
            raise ValueError(f"Maximize flags length ({len(self.objective_maximize)}) != objectives length ({len(objectives)})")
        
        # Load bigram position scores
        self.position_pair_scores = self._load_position_pair_scores(position_pair_score_table)
        
        # Load trigram position scores if provided
        self.position_triple_scores = {}
        if position_triple_score_table:
            self.position_triple_scores = self._load_position_triple_scores(position_triple_score_table)

        # Determine which objectives are trigram-based
        self.trigram_objectives = set(self.position_triple_scores.keys())
        self.bigram_objectives = set(obj for obj in objectives if obj not in self.trigram_objectives)

        if verbose:
            print(f"Initializing Extended WeightedMOOScorer:")
            print(f"  Objectives: {objectives}")
            print(f"  Items: {self.items}")
            print(f"  Positions: {self.positions}")
        else:
            print(f"Loading {len(self.bigram_objectives)} bigram objectives, {len(items)} items...")
            print(f"Loading {len(self.trigram_objectives)} trigram objectives, {len(items)} items...")

        # Load item pair/triple frequencies for weighting
        self.item_pair_scores = self._load_item_pair_scores(item_pair_score_table)
        self.use_bigram_weighting = len(self.item_pair_scores) > 0
        
        self.item_triple_scores = {}
        if item_triple_score_table:
            self.item_triple_scores = self._load_item_triple_scores(item_triple_score_table)
        self.use_trigram_weighting = len(self.item_triple_scores) > 0
          
        if self.use_bigram_weighting:
            bigram_total = sum(self.item_pair_scores.values())
            print(f"Bigram weighting: {len(self.item_pair_scores)} pairs, total score: {bigram_total:,.0f}")
        else:
            print(f"Using unweighted bigram scoring")
            
        if self.use_trigram_weighting:
            trigram_total = sum(self.item_triple_scores.values())
            print(f"Trigram weighting: {len(self.item_triple_scores)} triples, total score: {trigram_total:,.0f}")
        elif self.trigram_objectives:
            print(f"Using unweighted trigram scoring")

        # Create compatibility arrays for existing search infrastructure
        n_items, n_positions = len(self.items), len(self.positions)
        self.arrays = ScoringArrays(
            item_scores=np.ones(n_items, dtype=np.float32),
            item_pair_matrix=np.ones((n_items, n_items), dtype=np.float32),
            position_matrix=np.ones((n_positions, n_positions), dtype=np.float32)
        )

        # Precompute slot-pair score matrices for fast scoring path.
        # slot_pair_score[obj_idx, i, j] = penalised raw _pair_score for slots (i, j).
        self._bigram_obj_indices: List[int] = [
            i for i, obj in enumerate(self.objectives) if obj in self.bigram_objectives
        ]
        self._trigram_obj_indices: List[int] = [
            i for i, obj in enumerate(self.objectives) if obj in self.trigram_objectives
        ]

        if self._bigram_obj_indices:
            self._slot_pair_score = np.zeros(
                (len(self._bigram_obj_indices), n_positions, n_positions), dtype=np.float64
            )
            for k, obj_idx in enumerate(self._bigram_obj_indices):
                obj = self.objectives[obj_idx]
                ps = self.position_pair_scores[obj]
                for i in range(n_positions):
                    for j in range(n_positions):
                        if i == j:
                            continue
                        self._slot_pair_score[k, i, j] = self._pair_score(i, j, ps)
        else:
            self._slot_pair_score = None

        # Precompute item-pair weight matrix indexed by item indices.
        if self.use_bigram_weighting:
            self._item_pair_weight = np.zeros((n_items, n_items), dtype=np.float64)
            for i in range(n_items):
                for j in range(n_items):
                    if i == j:
                        continue
                    self._item_pair_weight[i, j] = self.item_pair_scores.get(
                        self.items[i] + self.items[j], 0.0
                    )
        else:
            self._item_pair_weight = None

    def _load_position_pair_scores(self, position_pair_score_table: str) -> Dict[str, Dict[str, float]]:
        """Load position-pair scores for bigram objectives from CSV table."""
        if not Path(position_pair_score_table).exists():
            raise FileNotFoundError(f"Position-pair table not found: {position_pair_score_table}")

        try:
            df = pd.read_csv(position_pair_score_table, dtype={'position_pair': str})
        except Exception as e:
            raise ValueError(f"Error reading position-pair table: {e}")

        if 'position_pair' not in df.columns:
            raise ValueError("Position-pair table must have 'position_pair' column")
        
        # Only load bigram objectives from this table
        available_objectives = [obj for obj in self.objectives if obj in df.columns]
        
        position_pair_scores = {}
        for obj in available_objectives:
            scores = {}
            valid_pairs = 0
            
            for _, row in df.iterrows():
                key_pair = str(row['position_pair']).strip("'\"")
                if len(key_pair) == 2 and not pd.isna(row[obj]):
                    scores[key_pair.upper()] = float(row[obj])
                    valid_pairs += 1
            
            position_pair_scores[obj] = scores
            print(f"    {obj}: {valid_pairs} position-pair scores loaded")
        
        return position_pair_scores

    def _load_position_triple_scores(self, position_triple_score_table: str) -> Dict[str, Dict[str, float]]:
        """Load position-triple scores for trigram objectives from CSV table."""
        if not Path(position_triple_score_table).exists():
            print(f"    Warning: Position-triple table not found: {position_triple_score_table}")
            return {}

        try:
            df = pd.read_csv(position_triple_score_table, dtype={'position_triple': str})
        except Exception as e:
            print(f"    Warning: Error reading position-triple table: {e}")
            return {}

        if 'position_triple' not in df.columns:
            print(f"    Warning: Position-triple table must have 'position_triple' column")
            return {}
        
        # Find trigram objectives in this table
        available_objectives = [obj for obj in self.objectives if obj in df.columns]
        
        position_triple_scores = {}
        for obj in available_objectives:
            scores = {}
            valid_triples = 0
            
            for _, row in df.iterrows():
                position_triple = str(row['position_triple']).strip("'\"")
                if len(position_triple) == 3 and not pd.isna(row[obj]):
                    scores[position_triple.upper()] = float(row[obj])
                    valid_triples += 1
            
            position_triple_scores[obj] = scores
            print(f"    {obj}: {valid_triples} position-triple scores loaded")
        
        return position_triple_scores

    def _load_item_pair_scores(self, item_pair_score_table: str) -> Dict[str, float]:
        """Load item-pair frequencies for bigram weighting."""
        if not Path(item_pair_score_table).exists():
            print(f"    Warning: Item-pair score file not found: {item_pair_score_table}")
            return {}
        
        try:
            df = pd.read_csv(item_pair_score_table)
        except Exception as e:
            print(f"    Warning: Error reading item-pair score file: {e}")
            return {}
        
        # Find appropriate columns
        item_pair_col = self._find_column(df, ['item_pair', 'pair', 'bigram', 'letter_pair'])
        freq_col = self._find_column(df, ['score', 'normalized_frequency', 'frequency'])
        
        if not item_pair_col or not freq_col:
            print(f"    Warning: Required columns not found in item-pair score file")
            return {}
        
        frequencies = {}
        for _, row in df.iterrows():
            item_pair = str(row[item_pair_col]).strip().upper()
            if len(item_pair) == 2:
                frequencies[item_pair] = float(row[freq_col])
        
        return frequencies

    def _load_item_triple_scores(self, item_triple_score_table: str) -> Dict[str, float]:
        """Load item-triple frequencies for trigram weighting."""
        if not Path(item_triple_score_table).exists():
            print(f"    Warning: Item-triple score file not found: {item_triple_score_table}")
            return {}
        
        try:
            df = pd.read_csv(item_triple_score_table)
        except Exception as e:
            print(f"    Warning: Error reading item-triple score file: {e}")
            return {}
        
        # Find appropriate columns
        item_triple_col = self._find_column(df, ['item_triple', 'triple', 'trigram', 'letter_triple'])
        freq_col = self._find_column(df, ['score', 'normalized_frequency', 'frequency'])
        
        if not item_triple_col or not freq_col:
            print(f"    Warning: Required columns not found in item-triple score file")
            return {}
        
        frequencies = {}
        for _, row in df.iterrows():
            item_triple = str(row[item_triple_col]).strip().upper()
            if len(item_triple) == 3:
                frequencies[item_triple] = float(row[freq_col])
        
        return frequencies
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column name from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _pair_score(self, slot_i_idx: int, slot_j_idx: int,
                    position_pair_scores: Dict[str, float]) -> float:
        """
        Score the position-pair contribution for two slots, summing over all
        constituent single-key pairs. Combos of size > 2 are penalised by
        combo_penalty^(max_size - 2). Same-finger constituent pairs are
        penalised by combo_same_finger_penalty^count, including:
          - in-slot SFBs (constituents within the same combo on the same finger)
          - cross-slot SFBs (a bigram transition between two slots where some
            constituent of one and some (different) constituent of the other
            share a finger). Cross-slot SFBs are only counted when at least
            one of the two slots is a combo, since single-vs-single SFBs are
            already encoded in the engram_same_finger position-pair table.
        """
        consts_i = self.position_constituents[slot_i_idx]
        consts_j = self.position_constituents[slot_j_idx]
        raw = 0.0
        for a in consts_i:
            for b in consts_j:
                if a != b:
                    raw += position_pair_scores.get((a + b).upper(), 0.0)
        max_size = max(len(consts_i), len(consts_j))
        if max_size > 2:
            raw *= self.combo_penalty ** (max_size - 2)
        sf_pairs = (
            self.position_same_finger_pairs[slot_i_idx]
            + self.position_same_finger_pairs[slot_j_idx]
            + int(self._cross_sf_pairs[slot_i_idx, slot_j_idx])
        )
        if sf_pairs > 0:
            raw *= self.combo_same_finger_penalty ** sf_pairs
        return raw

    def _triple_score(self, slot_i_idx: int, slot_j_idx: int, slot_k_idx: int,
                      position_triple_scores: Dict[str, float]) -> float:
        """
        Score the position-triple contribution for three slots, summing over
        all constituent single-key triples. Combos of size > 2 in any slot are
        penalised by combo_penalty^(max_size - 2). Same-finger constituent
        pairs (in-slot and cross-slot, when a combo is involved) are penalised
        by combo_same_finger_penalty^count.
        """
        consts_i = self.position_constituents[slot_i_idx]
        consts_j = self.position_constituents[slot_j_idx]
        consts_k = self.position_constituents[slot_k_idx]
        raw = 0.0
        for a in consts_i:
            for b in consts_j:
                if a == b:
                    continue
                for c in consts_k:
                    if c == a or c == b:
                        continue
                    raw += position_triple_scores.get((a + b + c).upper(), 0.0)
        max_size = max(len(consts_i), len(consts_j), len(consts_k))
        if max_size > 2:
            raw *= self.combo_penalty ** (max_size - 2)
        sf_pairs = (
            self.position_same_finger_pairs[slot_i_idx]
            + self.position_same_finger_pairs[slot_j_idx]
            + self.position_same_finger_pairs[slot_k_idx]
            + int(self._cross_sf_pairs[slot_i_idx, slot_j_idx])
            + int(self._cross_sf_pairs[slot_j_idx, slot_k_idx])
            + int(self._cross_sf_pairs[slot_i_idx, slot_k_idx])
        )
        if sf_pairs > 0:
            raw *= self.combo_same_finger_penalty ** sf_pairs
        return raw

    def score_layout_fast(self, mapping: np.ndarray) -> List[float]:
        """
        Vectorised scoring path used in the inner search loop.

        Equivalent to score_layout() for bigram objectives but uses precomputed
        slot-pair score matrices for an O(k^2) numpy reduction instead of an
        O(k^2) Python loop with per-pair constituent expansion.
        Falls back to score_layout() if any trigram objectives are configured.
        """
        # Trigram objectives are rare here; reuse the (slower but correct) path.
        if self._trigram_obj_indices:
            return self.score_layout(mapping)

        if self._slot_pair_score is None:
            return self.score_layout(mapping)

        # Indices of items currently placed and the slot indices they occupy.
        placed_item_idx = np.where(mapping >= 0)[0]
        if placed_item_idx.size < 2:
            return [0.0] * len(self.objectives)

        slot_idx = mapping[placed_item_idx].astype(np.int64)

        # slot_pair[a, b] = score for (slot at placed_item_idx[a], slot at placed_item_idx[b])
        # Shape: (n_obj_bigram, k, k)
        sub = self._slot_pair_score[:, slot_idx[:, None], slot_idx[None, :]]

        if self.use_bigram_weighting and self._item_pair_weight is not None:
            w = self._item_pair_weight[placed_item_idx[:, None], placed_item_idx[None, :]]
            mask = (w > 0) & (sub != 0.0)
            weighted = sub * w
            num = (weighted * mask).sum(axis=(1, 2))
            denom_each = (w * mask).sum(axis=(1, 2))
            with np.errstate(divide='ignore', invalid='ignore'):
                obj_scores = np.where(denom_each > 0, num / denom_each, 0.0)
        else:
            mask = (sub != 0.0)
            num = (sub * mask).sum(axis=(1, 2))
            denom = mask.sum(axis=(1, 2))
            with np.errstate(divide='ignore', invalid='ignore'):
                obj_scores = np.where(denom > 0, num / denom, 0.0)

        # Map per-bigram-objective scores back into full objective vector.
        result = [0.0] * len(self.objectives)
        for k, obj_idx in enumerate(self._bigram_obj_indices):
            result[obj_idx] = float(obj_scores[k])
        return result

    def score_layout(self, mapping: np.ndarray, return_components: bool = False) -> List[float]:
        """
        Score layout for all objectives using appropriate bigram/trigram scoring.

        Args:
            mapping: Array where mapping[i] = position_index for items[i] (-1 for unassigned)
            return_components: If True, return scores + combined average
            
        Returns:
            List of objective scores, optionally with combined average appended
        """
        scores = []
        
        for i, obj in enumerate(self.objectives):
            if obj in self.trigram_objectives:
                score = self._score_single_trigram_objective(mapping, obj)
            else:
                score = self._score_single_bigram_objective(mapping, obj)
            
            # Apply weights and direction transformations
            weighted_score = score * self.objective_weights[i]
            if not self.objective_maximize[i]:
                weighted_score = 1.0 - weighted_score
            
            scores.append(weighted_score)
        
        if return_components:
            combined_average = sum(scores) / len(scores) if scores else 0.0
            return scores + [combined_average]
        else:
            return scores
    
    def _score_single_bigram_objective(self, mapping: np.ndarray, objective: str) -> float:
        """Score layout for single bigram objective."""
        position_pair_scores = self.position_pair_scores[objective]
        
        # Get currently placed items and the slot indices they occupy.
        placed_items = []
        placed_slot_indices = []
        
        for i, pos_idx in enumerate(mapping):
            if pos_idx >= 0:
                placed_items.append(self.items[i])
                placed_slot_indices.append(int(pos_idx))
        
        if len(placed_items) < 2:
            return 0.0
        
        # Calculate score using bigram logic
        if self.use_bigram_weighting:
            return self._calculate_bigram_weighted_score(
                placed_items, placed_slot_indices, position_pair_scores)
        else:
            return self._calculate_bigram_unweighted_score(
                placed_items, placed_slot_indices, position_pair_scores)

    def _score_single_trigram_objective(self, mapping: np.ndarray, objective: str) -> float:
        """Score layout for single trigram objective."""
        position_triple_scores = self.position_triple_scores[objective]
        
        # Get currently placed items and the slot indices they occupy.
        placed_items = []
        placed_slot_indices = []
        
        for i, pos_idx in enumerate(mapping):
            if pos_idx >= 0:
                placed_items.append(self.items[i])
                placed_slot_indices.append(int(pos_idx))
        
        if len(placed_items) < 3:
            return 0.0
        
        # Calculate score using trigram logic
        if self.use_trigram_weighting:
            return self._calculate_trigram_weighted_score(
                placed_items, placed_slot_indices, position_triple_scores)
        else:
            return self._calculate_trigram_unweighted_score(
                placed_items, placed_slot_indices, position_triple_scores)
    
    def _calculate_bigram_weighted_score(self, items: List[str], slot_indices: List[int], 
                                        position_pair_scores: Dict[str, float]) -> float:
        """Calculate score using item-pair score weighting (combo-aware)."""
        weighted_total = 0.0
        item_pair_score_total = 0.0
        
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    letter_pair = items[i] + items[j]
                    item_pair_score = self.item_pair_scores.get(letter_pair, 0.0)
                    if item_pair_score > 0:
                        score = self._pair_score(slot_indices[i], slot_indices[j], position_pair_scores)
                        if score != 0.0:
                            weighted_total += score * item_pair_score
                            item_pair_score_total += item_pair_score
        
        return weighted_total / item_pair_score_total if item_pair_score_total > 0 else 0.0
    
    def _calculate_bigram_unweighted_score(self, items: List[str], slot_indices: List[int],
                                          position_pair_scores: Dict[str, float]) -> float:
        """Calculate score without item_pair_score weighting (combo-aware)."""
        total_score = 0.0
        pair_count = 0
        
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    score = self._pair_score(slot_indices[i], slot_indices[j], position_pair_scores)
                    if score != 0.0:
                        total_score += score
                        pair_count += 1
        
        return total_score / pair_count if pair_count > 0 else 0.0

    def _calculate_trigram_weighted_score(self, items: List[str], slot_indices: List[int], 
                                         position_triple_scores: Dict[str, float]) -> float:
        """Calculate trigram score using item-triple score weighting (combo-aware)."""
        weighted_total = 0.0
        item_triple_score_total = 0.0
        
        n = len(items)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:
                        letter_triple = items[i] + items[j] + items[k]
                        item_triple_score = self.item_triple_scores.get(letter_triple, 0.0)
                        if item_triple_score > 0:
                            score = self._triple_score(slot_indices[i], slot_indices[j], slot_indices[k], position_triple_scores)
                            if score != 0.0:
                                weighted_total += score * item_triple_score
                                item_triple_score_total += item_triple_score
        
        return weighted_total / item_triple_score_total if item_triple_score_total > 0 else 0.0

    def _calculate_trigram_unweighted_score(self, items: List[str], slot_indices: List[int],
                                           position_triple_scores: Dict[str, float]) -> float:
        """Calculate trigram score without item_triple_score weighting (combo-aware)."""
        total_score = 0.0
        triple_count = 0
        
        n = len(items)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:
                        score = self._triple_score(slot_indices[i], slot_indices[j], slot_indices[k], position_triple_scores)
                        if score != 0.0:
                            total_score += score
                            triple_count += 1
        
        return total_score / triple_count if triple_count > 0 else 0.0
    
    def get_objective_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about objective score ranges for analysis."""
        stats = {}
        
        # Bigram objective stats
        for obj, scores in self.position_pair_scores.items():
            if scores:
                values = list(scores.values())
                stats[obj] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'count': len(values),
                    'type': 'bigram'
                }
        
        # Trigram objective stats  
        for obj, scores in self.position_triple_scores.items():
            if scores:
                values = list(scores.values())
                stats[obj] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'count': len(values),
                    'type': 'trigram'
                }
        
        return stats
    
    def clear_cache(self):
        """Clear any caches (no caching in this implementation)."""
        pass


def validate_item_pair_scoring_consistency(items: str, positions: str, objectives: List[str],
                                         position_pair_score_table: str, item_pair_score_table: str,
                                         verbose: bool = False) -> Dict[str, float]:
    """
    Validate that WeightedMOOScorer produces consistent results.
    
    This function can be used to compare results with score_layouts.py
    or to test scorer behavior on known layouts.
    """
    # Create mapping from strings
    items_list = list(items.upper())
    positions_list = list(positions.upper())
    
    if len(items_list) != len(positions_list):
        raise ValueError(f"Items length ({len(items_list)}) != positions length ({len(positions_list)})")
    
    # Create scorer
    scorer = WeightedMOOScorer(
        objectives=objectives,
        position_pair_score_table=position_pair_score_table,
        items=items_list,
        positions=positions_list,
        item_pair_score_table=item_pair_score_table
    )
    
    # Create mapping array (complete layout)
    mapping = np.arange(len(items_list), dtype=np.int32)
    
    # Score layout
    scores = scorer.score_layout(mapping)
    
    if verbose:
        print(f"\nValidation Results:")
        print(f"  Layout: {items} -> {positions}")
        for i, (obj, score) in enumerate(zip(objectives, scores)):
            print(f"  {obj}: {score:.9f}")
    
    return dict(zip(objectives, scores))


if __name__ == "__main__":
    # Example usage and basic testing
    print("Testing WeightedMOOScorer...")
    
    # Test configuration
    test_objectives = ['engram_keys', 'engram_rows']
    test_items = ['e', 't', 'a', 'o']
    test_positions = ['F', 'D', 'S', 'J']
    
    try:
        scorer = WeightedMOOScorer(
            objectives=test_objectives,
            position_pair_score_table='input/engram_2key_scores.csv',
            items=test_items,
            positions=test_positions
        )
        
        # Test complete layout
        mapping = np.array([0, 1, 2, 3], dtype=np.int32)  # e->F, t->D, a->S, o->J
        scores = scorer.score_layout(mapping)
        
        print(f"\nTest Results:")
        print(f"  Layout: {test_items} -> {test_positions}")
        for obj, score in zip(test_objectives, scores):
            print(f"  {obj}: {score:.9f}")
        
        # Test objective statistics
        stats = scorer.get_objective_stats()
        print(f"\nObjective Statistics:")
        for obj, stat in stats.items():
            print(f"  {obj}: range [{stat['min']:.3f}, {stat['max']:.3f}], mean {stat['mean']:.3f}")
        
        print("\nWeightedMOOScorer test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure 'input/engram_2key_scores.csv' exists with required objectives.")