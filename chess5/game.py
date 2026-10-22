from __future__ import annotations

import uuid
from typing import Optional

from .board import BLACK, WHITE, Board
from .players import BasePlayer
from .recorder import GameRecorder

COLOR_NAMES = {WHITE: "White", BLACK: "Black"}


def play_game(
    white: BasePlayer,
    black: BasePlayer,
    verbose: bool = False,
    record: bool = False,
    game_id: Optional[str] = None,
    board: Optional[Board] = None,
) -> tuple[Board, Optional[GameRecorder]]:
    """Run one game to completion. Returns the final board and, if
    `record=True`, the GameRecorder holding the full trajectory.
    Pass `board` to start from a custom position instead of the opening."""
    board = board or Board()
    players = {WHITE: white, BLACK: black}
    recorder = None
    if record:
        recorder = GameRecorder(
            player_names=[white.name, black.name],
            game_id=game_id or str(uuid.uuid4()),
        )

    if verbose:
        print(board)

    while not board.is_game_over():
        mover = board.current_player
        move = players[mover].get_move(board)

        board_before = board.clone() if recorder else None
        board.make_move(move)

        if recorder:
            recorder.record_move(board_before, move, board)

        if verbose:
            print(f"\n{COLOR_NAMES[mover]} plays {move}")
            print(board)

    if recorder:
        recorder.finish(board)

    if verbose:
        print()
        print(describe_result(board, players))

    return board, recorder


def describe_result(board: Board, players: dict) -> str:
    if board.winner:
        return (f"Checkmate! {COLOR_NAMES[board.winner]} "
                f"({players[board.winner].name}) wins!")
    return f"Draw by {board.end_reason}."
