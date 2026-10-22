"""5x5 chess CLI: play a game, or generate self-play training data.

Examples:
    python main.py play                         # human vs human
    python main.py play --vs random             # you (White) vs random AI
    python main.py play --vs random --color black
    python main.py selfplay -n 500              # 500 random-vs-random games -> data/games.jsonl
    python main.py export-npz                   # flatten data/games.jsonl -> data/games.npz
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import time
from collections import Counter
from pathlib import Path

from chess5.game import COLOR_NAMES, play_game
from chess5.players import HumanPlayer, RandomPlayer, MinimaxPlayer, RandomOpeningPlayer
from chess5.recorder import export_to_numpy

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_JSONL = DATA_DIR / "games.jsonl"


def make_player(kind: str, rng: random.Random, depth: int = 3, random_moves: int = 0):
    """Build one player from a CLI name. Add your AI here once it exists."""
    if kind == "human":
        return HumanPlayer()
    elif kind == "random":
        return RandomPlayer(rng)
    elif kind == "minimax":
        inner = MinimaxPlayer(depth=depth)
    else: raise ValueError(f"unknown player kind: {kind}")
    if random_moves > 0:
        return RandomOpeningPlayer(inner, random_moves=random_moves, rng=rng)
    return inner


def cmd_play(args: argparse.Namespace) -> None:
    rng = random.Random()
    you = HumanPlayer()
    opponent = make_player(args.vs, rng)
    white, black = (you, opponent) if args.color == "white" else (opponent, you)
    _, recorder = play_game(white, black, verbose=True, record=args.record)
    if recorder:
        recorder.append_to_jsonl(DEFAULT_JSONL)
        print(f"\nGame recorded to {DEFAULT_JSONL}")


def _play_chunk(task):
    """Play a batch of games. Runs inside a worker process, so everything it
    needs arrives through `task` (Windows spawns fresh processes rather than
    forking, so nothing is inherited from the parent).

    Returns (json_line, result_label) pairs -- the parent does all file
    writing, which avoids several processes appending to one file at once."""
    n_games, white_kind, black_kind, depth, random_moves, seed = task

    # When a neural player joins self-play, call torch.set_num_threads(1)
    # here. Torch otherwise sizes its thread pool to the whole machine, so
    # N workers each spawn N threads and fight each other.

    # Each chunk MUST have its own seed. Minimax is deterministic, so with a
    # shared seed every worker would replay the identical set of games.
    rng = random.Random(seed)

    out = []
    for _ in range(n_games):
        white = make_player(white_kind, rng, depth, random_moves)
        black = make_player(black_kind, rng, depth, random_moves)
        board, recorder = play_game(white, black, record=True)
        if board.winner:
            label = f"{COLOR_NAMES[board.winner].lower()} wins"
        else:
            label = f"draw ({board.end_reason})"
        out.append((json.dumps(recorder.record.to_dict()), label))
    return out


def cmd_selfplay(args: argparse.Namespace) -> None:
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, args.num_games)

    print(f"{args.white} (White) vs {args.black} (Black) -> {args.out}")
    print(f"{args.num_games} games across {workers} worker{'s' if workers > 1 else ''}")

    # Split into many small chunks rather than one per worker: games vary in
    # length, so small chunks keep every core busy instead of leaving some
    # idle waiting on a straggler.
    chunk_size = max(1, min(10, args.num_games // (workers * 4)))
    chunks = []
    remaining = args.num_games
    while remaining > 0:
        chunks.append(min(chunk_size, remaining))
        remaining -= chunk_size

    # A fixed --seed keeps the whole run reproducible.
    base = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    tasks = [
        (n, args.white, args.black, args.depth, args.random_moves, base + i)
        for i, n in enumerate(chunks)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    done = 0
    results = Counter()
    report_every = max(1, args.num_games // 20)
    next_report = report_every

    with open(args.out, "a", encoding="utf-8") as f:
        if workers == 1:
            batches = map(_play_chunk, tasks)
        else:
            pool = multiprocessing.Pool(processes=workers)
            batches = pool.imap_unordered(_play_chunk, tasks)

        try:
            for batch in batches:
                for line, label in batch:
                    f.write(line + "\n")
                    results[label] += 1
                done += len(batch)
                if done >= next_report:
                    rate = done / max(1e-9, time.time() - start)
                    eta = (args.num_games - done) / rate
                    print(f"{done}/{args.num_games} games  ({rate:.1f}/s, ~{eta / 60:.1f} min left)")
                    next_report += report_every
        except KeyboardInterrupt:
            print("\ninterrupted -- games completed so far are saved")
            if workers > 1:
                pool.terminate()
            raise
        finally:
            if workers > 1:
                pool.close()
                pool.join()

    elapsed = time.time() - start
    print(f"Done. {done} games in {elapsed:.1f}s ({done / max(1e-9, elapsed):.1f}/s) -> {args.out}")
    print("Results: " + ", ".join(f"{k} {v}" for k, v in results.most_common()))


def cmd_export_npz(args: argparse.Namespace) -> None:
    export_to_numpy(args.jsonl, args.out)
    print(f"Exported {args.jsonl} -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="5x5 chess (Gardner's minichess)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_play = sub.add_parser("play", help="Play an interactive game")
    p_play.add_argument("--vs", choices=["human", "random", "minimax"], default="human")
    p_play.add_argument("--color", choices=["white", "black"], default="white",
                        help="Your side when playing against an AI")
    p_play.add_argument("--record", action="store_true", help="Save this game to data/games.jsonl")
    p_play.set_defaults(func=cmd_play)

    p_self = sub.add_parser("selfplay", help="Generate training data via self-play")
    p_self.add_argument("-n", "--num-games", type=int, default=100, dest="num_games")
    p_self.add_argument("--seed", type=int, default=None)
    p_self.add_argument("--white", choices=["random", "minimax"], default="random")
    p_self.add_argument("--black", choices=["random", "minimax"], default="random")
    p_self.add_argument("--out", type=Path, default=DEFAULT_JSONL)
    p_self.add_argument("--depth", type=int, default=3)
    p_self.add_argument("--random-moves", type=int, default=6, dest="random_moves")
    p_self.add_argument("--workers", type=int, default=0,
                        help="Parallel processes (0 = all cores but one, 1 = no multiprocessing)")
    p_self.set_defaults(func=cmd_selfplay)

    p_export = sub.add_parser("export-npz", help="Flatten a JSONL game log into a .npz tensor file")
    p_export.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    p_export.add_argument("--out", type=Path, default=DATA_DIR / "games.npz")
    p_export.set_defaults(func=cmd_export_npz)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
