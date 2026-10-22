import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess5.board import (
    BLACK, KING, NUM_ACTIONS, PAWN, QUEEN, WHITE, Board, Move, parse_square,
)


def mv(text: str) -> Move:
    return Move.from_uci(text)


def test_initial_state():
    b = Board()
    assert b.current_player == WHITE
    assert b.winner is None
    assert not b.is_game_over()
    # Five one-step pawn pushes plus Nb1-a3 and Nb1-c3. No double steps.
    assert sorted(m.to_uci() for m in b.legal_moves()) == [
        "a2a3", "b1a3", "b1c3", "b2b3", "c2c3", "d2d3", "e2e3",
    ]
    assert b.grid[4][4] == KING and b.grid[0][4] == -KING


def test_make_move_flips_player():
    b = Board()
    b.make_move(mv("c2c3"))
    assert b.grid[3][2] == 0 and b.grid[2][2] == PAWN
    assert b.current_player == BLACK
    assert b.move_count == 1


def test_illegal_move_raises():
    b = Board()
    try:
        b.make_move(mv("c2c4"))
        assert False, "expected ValueError for pawn double step"
    except ValueError:
        pass


def test_pinned_piece_cannot_leave_file():
    b = Board.from_string([
        "r...k",
        ".....",
        ".....",
        "R....",
        "K....",
    ])
    legal = b.legal_moves()
    assert mv("a2b2") not in legal
    assert mv("a2a5") in legal


def test_checkmate():
    b = Board.from_string([
        "k....",
        "..K..",
        ".....",
        ".....",
        ".Q...",
    ])
    b.make_move(mv("b1b4"))
    assert b.is_game_over()
    assert b.end_reason == "checkmate"
    assert b.winner == WHITE
    assert b.current_player == BLACK
    assert b.legal_moves() == []


def test_stalemate():
    b = Board.from_string([
        "k....",
        "..K..",
        "....Q",
        ".....",
        ".....",
    ])
    b.make_move(mv("e3b3"))
    assert b.end_reason == "stalemate"
    assert b.is_draw()
    assert b.winner is None


def test_promotion():
    b = Board.from_string([
        "....k",
        ".P...",
        ".....",
        ".....",
        "K....",
    ])
    promos = [m for m in b.legal_moves() if m.from_sq == parse_square("b4")]
    assert len(promos) == 4
    b.make_move(mv("b4b5q"))
    assert b.grid[0][1] == QUEEN


def test_insufficient_material():
    b = Board.from_string([
        "....k",
        ".....",
        "..N..",
        ".....",
        "K....",
    ])
    assert b.end_reason == "insufficient-material"
    assert b.is_draw()


def test_threefold_repetition():
    b = Board()
    shuffle = ["b1a3", "b5c3", "a3b1", "c3b5"]
    for text in shuffle * 2:
        assert not b.is_game_over()
        b.make_move(mv(text))
    assert b.end_reason == "repetition"


def test_clone_is_independent():
    b = Board()
    b.make_move(mv("c2c3"))
    c = b.clone()
    c.make_move(mv("b4c3"))
    assert b.grid != c.grid
    assert b.move_count == 1 and c.move_count == 2


def test_move_encodings_roundtrip():
    for text in ["a1a2", "e5a1", "b4b5q", "c2d1n"]:
        m = mv(text)
        assert m.to_uci() == text
        assert Move.from_index(m.to_index()) == m
        assert 0 <= m.to_index() < NUM_ACTIONS


def test_random_games_terminate():
    rng = random.Random(0)
    reasons = {"checkmate", "stalemate", "repetition", "fifty-move", "insufficient-material"}
    for _ in range(50):
        b = Board()
        while not b.is_game_over():
            b.make_move(rng.choice(b.legal_moves()))
        assert b.end_reason in reasons
