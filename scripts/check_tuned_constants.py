"""
check_tuned_constants.py

Assert that every tuned constant in src/optimiser.py and src/dc_predictor.py
has a corresponding evaluation document in docs/evaluations/.

A "tuned constant" is a value that was chosen to optimise model quality
rather than being derived from FPL rules or fit from data. Each such
constant must have a docs/evaluations/{slug}.md file explaining the value,
how it was chosen, and when/how to revisit it.

Run this in CI after any change to src/optimiser.py or src/dc_predictor.py:
    python scripts/check_tuned_constants.py

Exit 0 if all constants are documented; exit 1 with a list of missing docs.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "docs" / "evaluations"

# Registry of tuned constants.
# Each entry: (slug, "source:CONSTANT_NAME", short description)
# slug maps to docs/evaluations/{slug}.md
#
# FPL rules (BUDGET, MAX_PER_CLUB, SQUAD_SIZE, STARTING_XI, PT_* scoring
# constants) are NOT listed — they are derived from FPL's published rules
# and cannot be tuned.
#
# DC model parameters (alpha, beta, attack_i, defense_i, rho, home_adv)
# are NOT listed — they are fit from data by train_dc.py, not set manually.
TUNED_CONSTANTS = [
    (
        "bench_weight",
        "src/optimiser.py: BENCH_WEIGHT",
        "Multiplier on bench players' predicted points in the ILP objective.",
    ),
    (
        "min_qualifying_games",
        "src/optimiser.py: MIN_QUALIFYING_GAMES_3, MIN_QUALIFYING_GAMES_5",
        "Minimum appearances in last 3/5 GWs for a player to be eligible.",
    ),
    (
        "min_chance_of_playing",
        "src/optimiser.py: MIN_CHANCE_OF_PLAYING",
        "Minimum FPL chance_of_playing_next_round (%) for squad eligibility.",
    ),
    (
        "transfer_horizon_decay",
        "src/optimiser.py: _DECAY",
        "Per-GW discount factor for amortising transfer hits over multi-GW horizons.",
    ),
]


def main() -> None:
    missing = []
    for slug, source, desc in TUNED_CONSTANTS:
        doc = EVAL_DIR / f"{slug}.md"
        if not doc.exists():
            missing.append((slug, source, desc))

    if not missing:
        print(f"OK: all {len(TUNED_CONSTANTS)} tuned constants have evaluation docs.")
        sys.exit(0)

    print(f"MISSING evaluation docs for {len(missing)} tuned constant(s):\n")
    for slug, source, desc in missing:
        print(f"  {slug}.md")
        print(f"    Constant : {source}")
        print(f"    Meaning  : {desc}")
        print(f"    Fix      : create docs/evaluations/{slug}.md explaining the "
              f"value, how it was chosen, and when/how to revisit it.")
        print()
    sys.exit(1)


if __name__ == "__main__":
    main()
