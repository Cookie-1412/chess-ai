"""Report what's in your recorded game files, without training on them.

Examples:
    python stats.py                                  # all .jsonl in data/
    python stats.py data/games.jsonl
    python stats.py "data/selfplay_gen*.jsonl"
"""

import argparse
from collections import Counter

from chess5.board import BLACK, WHITE
from chess5.recorder import expand_paths, game_signature, load_games


def main():
    parser = argparse.ArgumentParser(description="Summarize recorded game data")
    parser.add_argument("paths", nargs="*", default=["data"],
                        help="Files, directories, or glob patterns (default: data/)")
    args = parser.parse_args()

    total_games = total_positions = 0
    all_signatures = set()
    results = Counter()
    end_reasons = Counter()
    lengths = []

    print(f"{'file':<38} {'games':>7} {'unique':>7} {'positions':>10}")
    print("-" * 66)

    for path in expand_paths(args.paths):
        games = list(load_games(path))
        sigs = {game_signature(g) for g in games}
        positions = sum(len(g["moves"]) for g in games)

        total_games += len(games)
        total_positions += positions
        all_signatures |= sigs
        for g in games:
            results[{WHITE: "white", BLACK: "black"}.get(g["winner"], "draw")] += 1
            end_reasons[g["end_reason"]] += 1
            lengths.append(len(g["moves"]))

        print(f"{str(path):<38} {len(games):>7} {len(sigs):>7} {positions:>10,}")

    print("-" * 66)
    print(f"{'TOTAL':<38} {total_games:>7} {len(all_signatures):>7} {total_positions:>10,}")
    print()

    if not total_games:
        return

    dupes = total_games - len(all_signatures)
    print(f"duplicates:  {dupes} ({100 * dupes / total_games:.0f}%)"
          + ("  -> worth running dedupe.py" if dupes > total_games * 0.05 else "  -> fine"))

    def pct(n):
        return f"{n} ({100 * n / total_games:.0f}%)"

    print(f"results:     white {pct(results['white'])}, black {pct(results['black'])}, "
          f"draw {pct(results['draw'])}")
    print("end reasons: " + ", ".join(f"{k} {pct(v)}" for k, v in end_reasons.most_common()))

    lengths.sort()
    print(f"game length: avg {sum(lengths) / len(lengths):.0f} plies, "
          f"median {lengths[len(lengths) // 2]}, min {lengths[0]}, max {lengths[-1]}")

    # A value network learns "who wins from here". If nearly every game ends
    # the same way, predicting that one answer scores well without learning
    # anything about the position.
    biggest = max(results.values()) / total_games
    if biggest > 0.8:
        print(f"\nwarning: {100 * biggest:.0f}% of games share one result -- the network "
              "can score well by always predicting it. Consider more varied openings.")


if __name__ == "__main__":
    main()
