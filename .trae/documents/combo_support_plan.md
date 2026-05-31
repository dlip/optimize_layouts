# Plan: Add Combo Support to Layout Optimizer

## Summary
Extend the optimizer so a "position" can be either a **single key** (e.g. `F`) or a **combo** (e.g. `F+J`, two or more keys pressed simultaneously to produce one letter). The flagship use case is a 20‑letter layout that uses **16 single keys (2 per finger) + 4 combos**. Combos are auto‑generated from adjacent same‑hand key pairs in `positions_to_assign`. Combo scores are derived by **summing** the constituent single‑key scores; combos of size > 2 receive a heavy penalty multiplier. The optimizer chooses freely which letters land on single keys vs combos.

## Current State Analysis

### How positions and scoring work today
- [config.yaml](file:///Users/dane.lipscombe/code/optimize_layouts/config.yaml) defines `optimization.positions_to_assign` as a string of single‑character position IDs (e.g. `"ASDFJKL;EI"`). Each character is one key.
- [config.py](file:///Users/dane.lipscombe/code/optimize_layouts/config.py#L41-L48) normalizes positions to `upper()` and treats each char as one position.
- [moo_scoring.py](file:///Users/dane.lipscombe/code/optimize_layouts/moo_scoring.py#L295-L317) scores a layout by iterating placed items and looking up `positions[i] + positions[j]` (a 2‑char key like `"FD"`) in `position_pair_scores`.
- [moo_search.py](file:///Users/dane.lipscombe/code/optimize_layouts/moo_search.py#L420-L463) builds `all_positions = positions_assigned + positions_to_assign` and uses `pos_idx` integer indices into that list — so internally a position is already just an opaque index, not a literal char. This makes adding multi‑char "combo" positions straightforward.
- The position‑pair scoring CSV [input/engram_2key_scores.csv](file:///Users/dane.lipscombe/code/optimize_layouts/input/engram_2key_scores.csv) keys are 2‑char strings of single positions.

### Key design implications
- Combos are *additional* logical position slots; they do **not** consume their underlying single keys (typing `F` alone and `F+J` together are different inputs). So all 16 single positions + N combos coexist; the optimizer assigns one letter per slot.
- Internally the search already uses integer indices, so we can extend the positions list with synthetic combo entries (e.g. `"FJ"`, `"DF"`) without touching the search loop — only **scoring lookup** and **input parsing** need changes.
- Output CSV currently does `''.join(mapping[item] for item in expected_order)` ([optimize_layouts.py L219](file:///Users/dane.lipscombe/code/optimize_layouts/optimize_layouts.py#L219)). Multi‑char combo IDs break the simple `''.join`; we need a separator for serialization.

## Proposed Changes

### Decision summary (from clarifications)
1. **Combo source**: auto‑generated from adjacent same‑hand key pairs in `positions_to_assign`.
2. **Combo scoring**: combo's intrinsic score and any bigram involving a combo = **sum** of constituent single‑key scores, with a heavy penalty multiplier for combos of size > 2.
3. **Letter placement**: optimizer searches over all letter→slot assignments freely.

### Assumptions & Decisions

- **Adjacency model**: a key‑map of single positions → `(hand, finger, row)` is needed. We hard‑code one for the standard QWERTY 32‑key layout (already implicit in [keyboards/layouts_filter_patterns.py L94](file:///Users/dane.lipscombe/code/optimize_layouts/keyboards/layouts_filter_patterns.py#L94)). Two positions are *adjacent same‑hand* if they share a hand and are on adjacent fingers OR same finger different row. We expose this map and adjacency rule in a new helper module.
- **Combo size**: `max_combo_size` config option (default `2`). Combos of size 3 are generated only if max_combo_size ≥ 3, and only when all pairwise members are mutually adjacent same‑hand.
- **Combo ID format**: use the concatenation of single‑position chars wrapped in brackets, e.g. `[FJ]`, `[SDF]`. Brackets disambiguate from single positions in serialized output. Internally, the position list element is just the bracketed string.
- **Combo scoring formula** (for objective `obj`):
  - Let `P(X)` = list of single positions making up slot `X` (single key: `[X]`; combo: list of constituents).
  - **Position‑pair score** for slot pair `(X, Y)` on objective `obj`:
    `raw = Σ over (a∈P(X), b∈P(Y), a≠b) position_pair_scores[obj][a+b]`
    `penalty = combo_penalty^(max(|P(X)|,|P(Y)|) − 2)` (default `combo_penalty = 0.5`, configurable)
    `score = raw * penalty`
    For X==Y (same slot, only happens when scoring a single letter not a bigram — not used in current scoring path), undefined.
  - This satisfies "sum of involved keys' scores" and "> 2 key combos heavily penalised" via the penalty exponent.
  - For trigram objectives, same idea but over `P(X) × P(Y) × P(Z)` triples.
- **Item‑pair weighting** (frequency table) is unchanged: the score is weighted by the bigram letter frequency, exactly as today.
- **Constraints**: existing `items_to_constrain` / `positions_to_constrain` continue to apply on the *single‑key* characters. Combos are not constrained by default. (Future extension: allow constraining combos.)
- **20‑letter target keyboard** (concrete example for verification): 16 single keys = `ASDFJKL;ZXCVNM,.` (2 per finger, home + bottom row) and 4 letters land on combos drawn from adjacent same‑hand pairs (e.g. `[SD]`, `[DF]`, `[KL]`, `[NM]`).

### File-by-file changes

#### 1. New file: `combos.py` (new module)
- Define `QWERTY_FINGER_MAP: Dict[str, Tuple[str, int, int]]` — char → (hand, finger_index, row_index) for the 32 QWERTY keys.
- `def is_adjacent_same_hand(p1, p2) -> bool` — two single positions are same‑hand and adjacent finger or same finger.
- `def generate_combos(positions: List[str], max_size: int = 2) -> List[Tuple[str, ...]]` — return all adjacent same‑hand combos of size 2..max_size from the given single positions. Each combo returned in canonical sorted order so `[DF]` and `[FD]` are the same combo.
- `def combo_id(constituents: Tuple[str, ...]) -> str` — return `"[DF]"` form.
- `def parse_slot(slot_id: str) -> Tuple[str, ...]` — given a slot (`"F"` or `"[DF]"`) return tuple of single‑position chars.

**Why**: isolates all combo geometry/labelling logic so `moo_scoring.py`, `moo_search.py`, and `config.py` stay focused on their concerns.

#### 2. `config.py`
- Add fields to `OptimizationConfig`:
  - `enable_combos: bool = False`
  - `max_combo_size: int = 2`
  - `combo_penalty: float = 0.5`  (multiplier per extra key beyond size 2)
- In `__post_init__`, **skip `.upper()` mutation if a positions string contains brackets** (we keep using single‑char letters in `positions_to_assign`; combos are derived, not user‑written).
- Add a derived property `OptimizationConfig.combo_slots: List[str]` that, when `enable_combos`, returns auto‑generated combo IDs by calling `combos.generate_combos(positions_to_assign, max_combo_size)`.
- Update `validate_config` to:
  - Require `len(items_to_assign) <= len(positions_to_assign) + len(combo_slots)` when combos enabled.
  - Warn if `enable_combos` true but no adjacent pairs available.

**Why**: minimal, additive config surface; backward compatible (defaults off).

#### 3. `config.yaml`
- Add commented example block:
  ```yaml
  optimization:
    enable_combos: true
    max_combo_size: 2
    combo_penalty: 0.5
    items_to_assign:    "etaoinsrhldcumfpgwy"   # 20 letters total
    positions_to_assign: "ASDFJKL;ZXCVNM,."     # 16 single keys, 2 per finger
  ```
- Document that combos are auto‑derived; user does not list them.

#### 4. `moo_scoring.py`
- Constructor `WeightedMOOScorer.__init__`:
  - Accept `positions` list that may now contain combo IDs (`"[DF]"`).
  - Accept new kwargs: `combo_penalty: float = 0.5`.
  - Pre‑parse each position into `self.position_constituents: List[Tuple[str, ...]]` via `combos.parse_slot`.
- Replace direct `key_pair = positions[i] + positions[j]` lookups with a new helper:
  ```python
  def _pair_score(self, slot_i_idx, slot_j_idx, position_pair_scores) -> float:
      consts_i = self.position_constituents[slot_i_idx]
      consts_j = self.position_constituents[slot_j_idx]
      raw = 0.0
      for a in consts_i:
          for b in consts_j:
              if a != b:
                  raw += position_pair_scores.get((a + b).upper(), 0.0)
      penalty = self.combo_penalty ** max(0, max(len(consts_i), len(consts_j)) - 2)
      return raw * penalty
  ```
- Update `_calculate_bigram_weighted_score`, `_calculate_bigram_unweighted_score`, and the trigram counterparts to call `_pair_score` (and a `_triple_score` analogue) using slot indices instead of string concatenation. The functions already receive `placed_positions` lists; switch them to receive `placed_position_indices` so we can index into `position_constituents`.
- Mirror the same change in [MOOUpperBoundCalculator._calculate_current_bigram_score](file:///Users/dane.lipscombe/code/optimize_layouts/moo_search.py#L346) so branch‑and‑bound bounds remain valid (sum across constituents inflates raw values; the existing `true_max_position_scores` upper‑bound logic must be multiplied by `max_combo_size^2` to remain a valid upper bound — implement this in `MOOUpperBoundCalculator.__init__` by reading scorer's `max_combo_size`).

**Why**: this is the structural change that makes a "position" polymorphic. All scoring math goes through `_pair_score` so combos and single keys are treated uniformly.

#### 5. `moo_search.py`
- In both `branch_bound_moo_search` and `exhaustive_moo_search`, after building `all_positions = positions_assigned + positions_to_assign`, append the auto‑generated combo IDs:
  ```python
  if config.optimization.enable_combos:
      from combos import generate_combos, combo_id
      combo_slots = [combo_id(c) for c in generate_combos(list(positions_to_assign), config.optimization.max_combo_size)]
      all_positions = all_positions + combo_slots
  ```
- Pass `combo_penalty` and `max_combo_size` through to the scorer in `optimize_layouts.run_moo_optimization`.
- In `MOOUpperBoundCalculator`, recompute `true_max_position_scores[obj] *= max_combo_size ** 2` so upper bounds stay valid (a combo↔combo pair can sum up to `max_combo_size²` constituent scores).
- No changes needed to the DFS state machine itself — combos are just extra position indices.

**Why**: search loop is already index‑based; only the candidate position list grows.

#### 6. `optimize_layouts.py`
- In `save_moo_results`, when joining positions use a comma separator if any slot is a combo:
  ```python
  if any(p.startswith('[') for p in mapping.values()):
      positions_str = ','.join(mapping[item] for item in expected_order)
  else:
      positions_str = ''.join(mapping[item] for item in expected_order)
  ```
- Pass `combo_penalty` and `max_combo_size` from config into `WeightedMOOScorer`.
- Update `validate_inputs` to count combo slots when checking `n_items <= n_positions`.

**Why**: keeps CSV output unambiguous when combos are present; preserves current single‑key output format when combos are off.

#### 7. `keyboards/display_layout.py` (visualization)
- Detect combo entries (bracketed) and render them as a small note under the keyboard ASCII (e.g. `Combos: [DF]→x, [KL]→y`), since combos cannot be drawn on the standard key grid.
- This is purely cosmetic; do not block the core optimization on it.

### Out of scope (explicitly not changed)
- Trigram input data files. Trigram scoring will work for combos via the same penalty formula but is only triggered if a trigram CSV is supplied.
- `keyboards/generate_configs_from_csv.py` and other batch scripts — leave untouched; users opt into combos via config.
- Constraint system changes for combos — left as future work.

## Verification Steps

1. **Unit‑level smoke test (no combos)**: run `python optimize_layouts.py --config config.yaml --dry-run` → must succeed unchanged (defaults `enable_combos: false`).
2. **Combo enumeration test**: in a Python REPL, `from combos import generate_combos; generate_combos(list("ASDFJKL;"), 2)` returns the expected adjacent same‑hand pairs (`SD`, `DF`, `KL`, `L;`, …).
3. **Scoring sanity**:
   - For two single positions, `_pair_score` must equal the direct CSV lookup (regression test).
   - For combo `[DF]` paired with single `J`, score equals `position_pair_scores["DJ"] + position_pair_scores["FJ"]`.
   - For combos `[DF]` and `[KL]`, score equals `DK+DL+FK+FL`.
   - For a 3‑key combo `[SDF]` paired with `J`, score equals `(SJ+DJ+FJ) * 0.5`.
4. **End‑to‑end 20‑letter run**:
   ```bash
   python optimize_layouts.py --config config.yaml \
     --objectives engram_key_preference,engram_row_separation \
     --search-mode branch-bound --time-limit 600
   ```
   with `enable_combos: true`, `items_to_assign` = 20 letters, `positions_to_assign` = 16 single keys. Verify:
   - Console reports auto‑generated combo count.
   - Output CSV contains rows where some letters are mapped to bracketed slots.
   - Pareto‑front validation still passes.
5. **Branch‑and‑bound correctness**: re‑run the same test with `--search-mode exhaustive` on a smaller (≤8 letter) instance and confirm both modes return the same Pareto front.

## Phase ordering for execution
1. Create `combos.py` and unit‑test `generate_combos` / `parse_slot`.
2. Extend `OptimizationConfig` and `config.yaml` (additive, default off).
3. Refactor `WeightedMOOScorer` scoring helpers to use slot indices + `_pair_score`.
4. Wire combo slot generation into `moo_search.py` and pass new params from `optimize_layouts.py`.
5. Update `MOOUpperBoundCalculator` upper bounds.
6. Adjust output serialization in `save_moo_results`.
7. Run the verification steps above.
