#!/usr/bin/env python3
"""
Thumb-combo (chord) support for layout optimization.

Model: a "thumb-combo" is the same single key X pressed simultaneously with a
dedicated thumb modifier key. It produces a different letter than bare X but
its physical typing characteristics (hand, finger, row, key-preference) are
identical to bare X — only an extra activation cost is incurred for engaging
the thumb modifier.

This module provides:
  - A hand/finger/row map for the standard 32-key QWERTY layout.
  - `generate_combos`, which produces one thumb-combo slot per single key.
  - Helpers to build/parse slot IDs.

Slot ID format:
  - Bare single-key slot:    "F"
  - Thumb-modified slot:     "[F]"
  In both cases the constituent tuple is ('F',). The bracket on the slot ID
  is the discriminator that the scorer uses to apply the thumb-combo penalty.
"""

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


def is_adjacent_same_hand(p1: str, p2: str) -> bool:
    """Retained for backwards compatibility; no longer used by `generate_combos`."""
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


def generate_combos(positions: List[str]) -> List[Tuple[str, ...]]:
    """
    Generate one thumb-combo slot per single key in `positions`.

    A thumb-combo is the same key X plus a dedicated thumb modifier; its
    typing characteristics (hand/finger/row/key-preference) are identical to
    bare X. The combo activation cost is captured by `combo_penalty` in the
    scorer, applied per thumb-combo slot involved in a bigram or trigram.

    Returns:
        List of size-1 tuples, e.g. ('F',), ('J',), ... — one per input
        position. Build the slot ID with `combo_id(('F',))` -> '[F]'.
    """
    upper_positions = [p.upper() for p in positions]
    seen = set()
    out: List[Tuple[str, ...]] = []
    for p in upper_positions:
        t = (p,)
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def combo_id(constituents: Tuple[str, ...]) -> str:
    """Return the bracketed slot ID, e.g. ('F',) -> '[F]'."""
    return '[' + ''.join(constituents) + ']'


def parse_slot(slot_id: str) -> Tuple[str, ...]:
    """
    Parse a slot ID into its constituent single-key chars.

    'F'    -> ('F',)
    '[F]'  -> ('F',)
    """
    s = slot_id.strip()
    if s.startswith('[') and s.endswith(']'):
        return tuple(s[1:-1])
    return (s,)


def is_combo(slot_id: str) -> bool:
    """True if slot_id is a thumb-combo (bracketed)."""
    return slot_id.startswith('[') and slot_id.endswith(']')


def count_same_finger_pairs(constituents: Tuple[str, ...]) -> int:
    """
    Retained for backwards compatibility. With size-1 thumb-combos this
    always returns 0 (no in-slot pairs).
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
    Retained for backwards compatibility. With size-1 thumb-combos the
    scorer skips single-vs-single (already encoded in engram_same_finger
    table), so this returns 0 in that path.
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
    combos = generate_combos(test_positions)
    print(f"Thumb-combo slots for {test_positions}:")
    for c in combos:
        print(f"  {combo_id(c)}  ({c})")
    print(f"Total: {len(combos)} thumb-combo slots (one per single key)")
