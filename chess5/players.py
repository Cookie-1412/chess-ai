"""Player interfaces. Any AI you write just needs to implement
`get_move(board) -> Move` and can drop straight into `play_game`."""

from __future__ import annotations

import random, math
from abc import ABC, abstractmethod

from .board import EMPTY, QUEEN, WHITE, Board, Move
from .heuristics import score_position

# What _minimax returns for a decided position. Much bigger than any
# material score, so a forced mate always beats winning a queen.
TERMINAL_SCORE = 1000.0


class BasePlayer(ABC):
    name: str = "player"

    @abstractmethod
    def get_move(self, board: Board) -> Move:
        """Return one of board.legal_moves() given the current board state."""
        raise NotImplementedError


class HumanPlayer(BasePlayer):
    name = "human"

    def get_move(self, board: Board) -> Move:
        legal = board.legal_moves()
        color = "White" if board.current_player == WHITE else "Black"
        check = " (in check)" if board.is_in_check() else ""
        while True:
            raw = input(f"{color} to move{check}. Enter a move like b2b3, or 'moves': ")
            raw = raw.strip().lower().replace(" ", "").replace("-", "")
            if raw in ("moves", "list", "?"):
                print(" ".join(sorted(m.to_uci() for m in legal)))
                continue
            try:
                move = Move.from_uci(raw)
            except ValueError:
                print("Couldn't read that. Type the from-square then the to-square, like b2b3.")
                continue
            # Typing a promotion without a piece letter means queen.
            if move.promotion == EMPTY and Move(move.from_sq, move.to_sq, QUEEN) in legal:
                move = Move(move.from_sq, move.to_sq, QUEEN)
            if move not in legal:
                print("That move isn't legal here. Type 'moves' to see the options.")
                continue
            return move


class RandomPlayer(BasePlayer):
    """Picks uniformly among legal moves. Useful as a baseline opponent
    and for generating self-play training data cheaply."""

    name = "random"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def get_move(self, board: Board) -> Move:
        return self.rng.choice(board.legal_moves())


class MinimaxPlayer(BasePlayer):
    name = "minimax"

    def __init__(self, depth: int = 3):
        self.depth = depth

    def get_move(self, board: Board) -> Move:
        perspective_player = board.current_player
        best_score = -math.inf
        best_move = board.legal_moves()[0]
        for move in board.legal_moves():
            clone = board.clone()
            clone.make_move(move)
            score = self._minimax(clone, self.depth - 1, False, perspective_player)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

    def _minimax(self, board: Board, depth: int, maximizing: bool, perspective_player: int) -> float:
        # Base cases
        if board.is_game_over():
            if board.winner == perspective_player: return TERMINAL_SCORE + depth    # Win (sooner = higher)
            if board.winner is not None: return -(TERMINAL_SCORE + depth)           # Lose (sooner = lower)
            return 0.0                                                              # Draw
        if depth <= 0: return score_position(board, perspective_player)             # Out of budget

        # Recursion
        max_score = -math.inf
        min_score = math.inf
        for move in board.legal_moves():
            clone = board.clone()
            clone.make_move(move)
            score = self._minimax(clone, depth - 1, not maximizing, perspective_player)
            if maximizing and score > max_score:
                max_score = score
            if not maximizing and score < min_score:
                min_score = score

        if maximizing: return max_score
        return min_score


class RandomOpeningPlayer(BasePlayer):
    def __init__(self, inner: BasePlayer, random_moves: int = 6, rng: random.Random | None = None):
        self.inner = inner
        self.random_moves = random_moves
        self.rng = rng or random.Random()
        self.name = f"random-opening({inner.name})"

    def get_move(self, board: Board) -> Move:
        if board.move_count < self.random_moves:
            return self.rng.choice(board.legal_moves())
        return self.inner.get_move(board)