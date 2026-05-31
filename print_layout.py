"""Print the rank-1 layout from the latest results CSV as ASCII art."""
import csv
import glob
import os
import sys

# Find the latest result CSV (or accept one as argv).
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    candidates = sorted(glob.glob("output/layouts/moo_results_*.csv"), key=os.path.getmtime)
    if not candidates:
        sys.exit("No result CSV found in output/layouts/")
    csv_path = candidates[-1]

# Load rank requested (default rank 1).
target_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 1
with open(csv_path) as f:
    r = csv.DictReader(f)
    row = None
    for r_ in r:
        if int(r_["rank"]) == target_rank:
            row = r_
            break
if row is None:
    sys.exit(f"Rank {target_rank} not found in {csv_path}")

items = row["items"]
positions_field = row["positions"]
positions = [p.strip() for p in positions_field.split(",")]
mapping = dict(zip(items, positions))

scores = ", ".join(f"{k}={row[k]}" for k in [
    "engram_key_preference", "engram_row_separation",
    "engram_same_row", "engram_same_finger",
] if k in row)
print(f"File:  {os.path.basename(csv_path)}")
print(f"Rank:  {row['rank']}")
print(f"Score: {scores}")
print()

QWERTY_TOP  = list("QWERTYUIOP")
QWERTY_HOME = list("ASDFGHJKL;")

bare  = {p: "." for p in QWERTY_TOP + QWERTY_HOME}
combo = {p: "." for p in QWERTY_TOP + QWERTY_HOME}

for letter, slot in mapping.items():
    if slot.startswith("[") and slot.endswith("]"):
        combo[slot[1:-1]] = letter
    else:
        bare[slot] = letter

def pad(s):
    return f" {s.upper()} " if s != "." else " . "

def fmt_row(qrow, table):
    left  = "".join(f"|{pad(table[k])}" for k in qrow[:5]) + "|"
    right = "".join(f"|{pad(table[k])}" for k in qrow[5:]) + "|"
    return left + "   " + right

bar = "+---+---+---+---+---+   +---+---+---+---+---+"
print("=== BASE LAYER (bare keys) ===")
print(bar); print(fmt_row(QWERTY_TOP,  bare))
print(bar); print(fmt_row(QWERTY_HOME, bare))
print(bar)
print()
print("=== THUMB-COMBO LAYER (hold thumb modifier) ===")
print(bar); print(fmt_row(QWERTY_TOP,  combo))
print(bar); print(fmt_row(QWERTY_HOME, combo))
print(bar)
print()
print("Letter -> slot:")
for letter in sorted(mapping.keys()):
    print(f"  {letter} -> {mapping[letter]}")
