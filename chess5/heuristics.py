from __future__ import annotations

from .board import BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, Board, owner

PIECE_VALUES = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 0}

def score_position(board: Board, player: int) -> float:
    score = 0
    for row in board.grid:
        for col in row:
            if col == 0:
                continue
            value = PIECE_VALUES[abs(col)]  # Use abs() since opponent's pieces are negatives
            if owner(col) == player:
                score += value
            else:
                score -= value
    return score
