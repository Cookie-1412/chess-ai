"""How long does one MinimaxPlayer move take at each depth?

Run this before picking --depth for selfplay: the cost per ply multiplies,
so one extra level of depth can turn minutes of self-play into hours.
"""

import time

from chess5.board import Board
from chess5.players import MinimaxPlayer

# Stop going deeper once a single move takes longer than this.
TIME_LIMIT = 30.0

POSITIONS = {
    # Only 7 legal moves, so this flatters the search.
    "opening": Board(),
    # Open lines and 17 legal moves: closer to what a mid-game search faces.
    "middlegame": Board.from_string([
        "rnbqk",
        "p.p.p",
        ".p.p.",
        "P.P.P",
        "RNBQK",
    ]),
}

for label, board in POSITIONS.items():
    print(f"{label} ({len(board.legal_moves())} legal moves)", flush=True)
    previous = None
    for depth in range(1, 8):
        ai = MinimaxPlayer(depth=depth)
        start = time.time()
        move = ai.get_move(board)
        elapsed = time.time() - start
        print(f"  depth={depth}: {elapsed:.3f}s (chose {move})", flush=True)

        # Each extra ply multiplies the cost by roughly the same factor, so
        # guess the next depth's time before running it. Waiting until a depth
        # has already blown the limit could mean waiting 20x longer than it.
        if previous and previous > 0.01:
            predicted = elapsed * (elapsed / previous)
            if predicted > TIME_LIMIT:
                print(f"  stopping: depth={depth + 1} would take ~{predicted:.0f}s", flush=True)
                break
        previous = elapsed
    print()
