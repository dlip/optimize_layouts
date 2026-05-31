#!/usr/bin/env python3
"""
Combo (chord) support for layout optimization.

A "combo" is two or more keys pressed simultaneously to produce one letter.
This module provides:
  - A hand/finger/row map for the standard 32-key QWERTY layout.
  - Adjacency rules used to generate plausible combos.
  - Helpers to enumerate combo slots and to parse slot IDs.

Slot ID format:
  - Single-key slot: the bare key char, e.g. "F".
  - Combo slot:      bracketed concatenation of constituent chars in canonical
                     (sorted-by-QWERTY-position) order, e.g. "[DF]", "[SDF]".
"""

from itertools import combinations
from typing import Dict, List, Tuple

# Canonical QWERTY 32-key order (matches keyboards/layouts_filter_patterns.py)
QWERTY_ORDER = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"

# Map each char -> (hand, finger_index, row_index)
#   hand:   'L' or 'R'
#   finger: 0=pinky .. 3=index for each hand
#   row:    0=top, 1=home, 2=bottom
QWERTY_FINGER_MAP: Dict[str, Tuple[str, int, int]] = {
    # Top row
    'Q': ('L', 0, 0), 'W': ('L', 1, 0), 'E': ('L', 2, 0), 'R': ('L', 3, 0), 'T': ('L', 3, 0),
    'Y': ('R', 3, 0), 'U': ('R', 3, 0), 'I': ('R', 2, 0), 'O': ('R', 1, 0), 'P': ('R', 0, 0),
    '[': ('R', 0, 0),
    # Home row
    'A': ('L', 0, 1), 'S': ('L', 1, 1), 'D': ('L', 2, 1), 'F': ('L', 3, 1), 'G': ('L', 3, 1),
    'H': ('R', 3, 1), 'J': ('R', 3, 1), 'K': ('R', 2, 1), 'L': ('R', 1, 1), ';': ('R', 0, 1),
    "'": ('R', 0, 1),
    # Bottom row
    'Z': ('L', 0, 2), 'X': ('L', 1, 2), 'C': ('L', 2, 2), 'V': ('L', 3, 2), 'B': ('L', 3, 2),
    'N': ('R', 3, 2), 'M': ('R', 3, 2), ',': ('R', 2, 2), '.': ('R', 1, 2), '/': ('R', 0, 2),
}


def _qwerty_rank(ch: str) -> int:
    """Stable ordering of single-key chars by QWERTY position (for canonical combo IDs)."""
    try:
        return QWERTY_ORDER.index(ch)
    except ValueError:
        return len(QWERTY_ORDER) + ord(ch)


def is_adjacent_same_hand(p1: str, p2: str) -> bool:
    """
    Two single-key positions are 'adjacent same-hand' iff they share a hand and
    their fingers are the same or differ by 1 (i.e. neighbouring fingers).
    """
    if p1 == p2:
        return False
    info1 = QWERTY_FINGER_MAP.get(p1.upper())
    info2 = QWERTY_FINGER_MAP.get(p2.upper())
    if info1 is None or info2 is None:
        return False
    h1, f1, _r1 = info1
    h2, f2, _r2 = info2
    if h1 != h2:
        return False
    return abs(f1 - f2) <= 1


def _all_pairwise_adjacent(group: Tuple[str, ...]) -> bool:
    """Return True if every pair in `group` is adjacent same-hand."""
    for a, b in combinations(group, 2):
        if not is_adjacent_same_hand(a, b):
            return False
    return True


def generate_combos(positions: List[str], max_size: int = 2) -> List[Tuple[str, ...]]:
    """
    Enumerate all unique combo groups of size 2..max_size from `positions`.

    No adjacency or same-hand restriction is imposed — every C(N, k) unique
    combination is produced. Awkward combos (e.g. cross-hand or same-finger)
    are expected to be discouraged by `combo_same_finger_penalty` and the
    underlying position-pair scoring tables rather than filtered out here.

    Args:
        positions: list of single-key position chars
        max_size: maximum combo size (>=2)

    Returns:
        List of tuples, each tuple in canonical (QWERTY-rank) order.
        Duplicates removed.
    """
    if max_size < 2:
        return []

    upper_positions = [p.upper() for p in positions]
    combos_out: List[Tuple[str, ...]] = []
    seen = set()

    for size in range(2, max_size + 1):
        for group in combinations(upper_positions, size):
            canonical = tuple(sorted(group, key=_qwerty_rank))
            if canonical in seen:
                continue
            seen.add(canonical)
            combos_out.append(canonical)

    return combos_out


def combo_id(constituents: Tuple[str, ...]) -> str:
    """Return the bracketed slot ID, e.g. ('D','F') -> '[DF]'."""
    return '[' + ''.join(constituents) + ']'


def parse_slot(slot_id: str) -> Tuple[str, ...]:
    """
    Parse a slot ID into its constituent single-key chars.

    'F'    -> ('F',)
    '[DF]' -> ('D', 'F')
    """
    s = slot_id.strip()
    if s.startswith('[') and s.endswith(']'):
        return tuple(s[1:-1])
    return (s,)


def is_combo(slot_id: str) -> bool:
    """True if slot_id is a combo (bracketed)."""
    return slot_id.startswith('[') and slot_id.endswith(']')


def count_same_finger_pairs(constituents: Tuple[str, ...]) -> int:
    """
    Count how many unordered constituent pairs share the same (hand, finger).
    A 2-key combo on the same finger returns 1; a 3-key combo where all three
    share a finger returns 3 (C(3,2)=3); none-shared returns 0.
    """
    n = len(constituents)
    if n < 2:
        return 0
    count = 0
    for i in range(n):
        info_i = QWERTY_FINGER_MAP.get(constituents[i].upper())
        if info_i is None:
            continue
        for j in range(i + 1, n):
            info_j = QWERTY_FINGER_MAP.get(constituents[j].upper())
            if info_j is None:
                continue
            if info_i[0] == info_j[0] and info_i[1] == info_j[1]:
                count += 1
    return count


def count_cross_same_finger_pairs(consts_a: Tuple[str, ...], consts_b: Tuple[str, ...]) -> int:
    """
    Count how many cross-slot constituent pairs (one from consts_a, one from
    consts_b) share the same (hand, finger), excluding pairs where the two
    constituents are the SAME key (a == b). Used to detect "combo SFBs":
    bigram transitions in/out of a combo where the moving finger is the same.
    """
    count = 0
    for a in consts_a:
        info_a = QWERTY_FINGER_MAP.get(a.upper())
        if info_a is None:
            continue
        for b in consts_b:
            if a == b:
                continue
            info_b = QWERTY_FINGER_MAP.get(b.upper())
            if info_b is None:
                continue
            if info_a[0] == info_b[0] and info_a[1] == info_b[1]:
                count += 1
    return count


if __name__ == "__main__":
    # Quick sanity demo
    test_positions = list("ASDFJKL;")
    print(f"Adjacency demo for {test_positions}:")
    pairs = generate_combos(test_positions, max_size=2)
    for c in pairs:
        print(f"  {combo_id(c)}  ({c})")
    print(f"Total: {len(pairs)} adjacent same-hand pairs")
