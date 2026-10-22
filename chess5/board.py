"""Pure 5x5 chess engine: no I/O, no printing, no randomness.

Kept dependency-free and side-effect-free on purpose so the exact same
Board class can drive a CLI game, a batch of self-play games for
generating training data, or a future AI agent's search/inference loop.

Rules are Gardner's minichess (1969), the standard 5x5 variant:
  - Each side starts with R N B Q K on its back rank and five pawns.
  - Pieces move exactly as in normal chess.
  - Pawns only ever move one square, so there is no en passant.
  - No castling.
  - A pawn reaching the last rank promotes to a queen, rook, bishop or knight.
  - Checkmate wins. Stalemate, threefold repetition, the fifty-move rule
    and insufficient material are draws.

Board layout: grid[0] is rank 5 (Black's back rank) and grid[4] is rank 1
(White's back rank), so printing the grid top to bottom shows the board
from White's side. Squares are numbered 0..24 as row * 5 + col, so
a5 = 0, e5 = 4, a1 = 20, e1 = 24.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

SIZE = 5
NUM_SQUARES = SIZE * SIZE

EMPTY = 0
WHITE = 1
BLACK = 2

Player = int  # WHITE or BLACK

# Piece kinds. A cell holds +kind for a White piece and -kind for a Black
# one, so `cell > 0` means White, `cell < 0` means Black, abs(cell) is the
# kind. Handy for a network too: one signed int per square.
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

PIECE_LETTERS = {PAWN: "p", KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q", KING: "k"}
LETTER_PIECES = {letter: kind for kind, letter in PIECE_LETTERS.items()}

PROMOTION_PIECES = (QUEEN, ROOK, BISHOP, KNIGHT)

NO_PROGRESS_LIMIT = 100  # plies without a capture or pawn move (the fifty-move rule)
REPETITION_LIMIT = 3

# Rows top to bottom (rank 5 first). Uppercase = White, lowercase = Black.
START_POSITION = (
    "rnbqk",
    "ppppp",
    ".....",
    "PPPPP",
    "RNBQK",
)

KNIGHT_STEPS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
KING_STEPS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
BISHOP_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ROOK_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SLIDER_DIRS = {BISHOP: BISHOP_DIRS, ROOK: ROOK_DIRS, QUEEN: BISHOP_DIRS + ROOK_DIRS}


def other_player(player: Player) -> Player:
    return BLACK if player == WHITE else WHITE


def player_sign(player: Player) -> int:
    return 1 if player == WHITE else -1


def owner(cell: int) -> Optional[Player]:
    if cell > 0:
        return WHITE
    if cell < 0:
        return BLACK
    return None


def square_name(sq: int) -> str:
    row, col = divmod(sq, SIZE)
    return f"{'abcde'[col]}{SIZE - row}"


def parse_square(name: str) -> int:
    if len(name) != 2 or name[0] not in "abcde" or name[1] not in "12345":
        raise ValueError(f"bad square: {name!r}")
    col = "abcde".index(name[0])
    row = SIZE - int(name[1])
    return row * SIZE + col


# Promotion slot 0 means "no promotion", so every index decodes to exactly
# one move. Most of the 3125 indices are never legal; a policy head just
# masks them out.
_PROMOTION_SLOTS = (EMPTY,) + PROMOTION_PIECES
NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES * len(_PROMOTION_SLOTS)


class Move(NamedTuple):
    from_sq: int
    to_sq: int
    promotion: int = EMPTY  # piece kind (QUEEN etc.) when a pawn promotes

    def to_uci(self) -> str:
        """e.g. 'b2b3', or 'b4b5q' for a promotion."""
        text = square_name(self.from_sq) + square_name(self.to_sq)
        if self.promotion:
            text += PIECE_LETTERS[self.promotion]
        return text

    @classmethod
    def from_uci(cls, text: str) -> "Move":
        text = text.strip().lower()
        if len(text) not in (4, 5):
            raise ValueError(f"bad move: {text!r}")
        promotion = EMPTY
        if len(text) == 5:
            promotion = LETTER_PIECES.get(text[4], EMPTY)
            if promotion not in PROMOTION_PIECES:
                raise ValueError(f"bad promotion piece: {text[4]!r}")
        return cls(parse_square(text[:2]), parse_square(text[2:4]), promotion)

    def to_index(self) -> int:
        """Unique int in [0, NUM_ACTIONS) for use as a policy-network action."""
        slot = _PROMOTION_SLOTS.index(self.promotion)
        return (slot * NUM_SQUARES + self.from_sq) * NUM_SQUARES + self.to_sq

    @classmethod
    def from_index(cls, index: int) -> "Move":
        rest, to_sq = divmod(index, NUM_SQUARES)
        slot, from_sq = divmod(rest, NUM_SQUARES)
        return cls(from_sq, to_sq, _PROMOTION_SLOTS[slot])

    def __str__(self) -> str:
        return self.to_uci()


PositionKey = Tuple[Tuple[int, ...], Player]


@dataclass
class Board:
    grid: List[List[int]] = field(default=None)  # type: ignore[assignment]
    current_player: Player = WHITE
    winner: Optional[Player] = None
    # Why the game ended: "checkmate", "stalemate", "repetition",
    # "fifty-move" or "insufficient-material". None while still playing.
    end_reason: Optional[str] = None
    last_move: Optional[Move] = None
    move_count: int = 0
    halfmove_clock: int = 0  # plies since the last capture or pawn move
    # Position counts since the last capture or pawn move. Earlier
    # positions can never come back, so this stays small.
    history: Dict[PositionKey, int] = field(default=None, repr=False)  # type: ignore[assignment]
    _legal: List[Move] = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self):
        if self.grid is None:
            self.grid = _parse_rows(START_POSITION)
        if self.history is None:
            self.history = {self.position_key(): 1}
        if self._legal is None:
            self._update_status()

    @classmethod
    def from_string(cls, rows: Sequence[str], current_player: Player = WHITE) -> "Board":
        """Build any position from 5 row strings, rank 5 first, e.g.
        Board.from_string(["k....", ".....", "..K..", ".....", "....Q"]).
        Useful for testing your AI on specific puzzles."""
        return cls(grid=_parse_rows(rows), current_player=current_player)

    def clone(self) -> "Board":
        return Board(
            grid=[row[:] for row in self.grid],
            current_player=self.current_player,
            winner=self.winner,
            end_reason=self.end_reason,
            last_move=self.last_move,
            move_count=self.move_count,
            halfmove_clock=self.halfmove_clock,
            history=dict(self.history),
            _legal=list(self._legal),
        )

    def legal_moves(self) -> List[Move]:
        if self.is_game_over():
            return []
        return list(self._legal)

    def is_game_over(self) -> bool:
        return self.end_reason is not None

    def is_draw(self) -> bool:
        return self.end_reason is not None and self.winner is None

    def is_in_check(self, player: Optional[Player] = None) -> bool:
        player = player or self.current_player
        king_sq = self._find_king(player)
        return king_sq is not None and self._is_attacked(king_sq, other_player(player))

    def piece_at(self, sq: int) -> int:
        return self.grid[sq // SIZE][sq % SIZE]

    def make_move(self, move: Move) -> int:
        """Play `move` for the current player. Returns the captured cell
        value (EMPTY if nothing was captured). Mutates the board in place
        and always flips current_player, so after checkmate current_player
        is the side that got mated and `winner` is the other side."""
        if self.is_game_over():
            raise ValueError("game is already over")
        if move not in self._legal:
            raise ValueError(f"illegal move: {move.to_uci()}")

        fr, fc = divmod(move.from_sq, SIZE)
        tr, tc = divmod(move.to_sq, SIZE)
        piece = self.grid[fr][fc]
        captured = self.grid[tr][tc]

        if move.promotion:
            piece = move.promotion * player_sign(self.current_player)
        self.grid[tr][tc] = piece
        self.grid[fr][fc] = EMPTY

        if abs(piece) == PAWN or move.promotion or captured != EMPTY:
            self.halfmove_clock = 0
            self.history.clear()
        else:
            self.halfmove_clock += 1

        self.current_player = other_player(self.current_player)
        self.last_move = move
        self.move_count += 1

        key = self.position_key()
        self.history[key] = self.history.get(key, 0) + 1

        self._update_status()
        return captured

    def position_key(self) -> PositionKey:
        """Hashable snapshot of the position (pieces + side to move). Good
        as a transposition-table key for your search."""
        return tuple(cell for row in self.grid for cell in row), self.current_player

    def to_array(self) -> List[List[int]]:
        """Plain nested-list snapshot of the board, safe to json.dumps
        or feed into numpy.array()/torch.tensor() for training."""
        return [row[:] for row in self.grid]

    def _update_status(self) -> None:
        self._legal = self._generate_legal_moves(self.current_player)
        if not self._legal:
            if self.is_in_check():
                self.winner = other_player(self.current_player)
                self.end_reason = "checkmate"
            else:
                self.end_reason = "stalemate"
        elif self._insufficient_material():
            self.end_reason = "insufficient-material"
        elif self.history.get(self.position_key(), 0) >= REPETITION_LIMIT:
            self.end_reason = "repetition"
        elif self.halfmove_clock >= NO_PROGRESS_LIMIT:
            self.end_reason = "fifty-move"

    def _generate_legal_moves(self, player: Player) -> List[Move]:
        king_sq = self._find_king(player)
        enemy = other_player(player)
        legal = []
        for move in self._pseudo_legal_moves(player):
            # Try the move on the grid, see if our king is attacked, undo.
            fr, fc = divmod(move.from_sq, SIZE)
            tr, tc = divmod(move.to_sq, SIZE)
            moving, target = self.grid[fr][fc], self.grid[tr][tc]
            self.grid[tr][tc], self.grid[fr][fc] = moving, EMPTY
            sq = move.to_sq if abs(moving) == KING else king_sq
            safe = sq is None or not self._is_attacked(sq, enemy)
            self.grid[fr][fc], self.grid[tr][tc] = moving, target
            if safe:
                legal.append(move)
        return legal

    def _pseudo_legal_moves(self, player: Player) -> List[Move]:
        """Moves that follow piece movement rules but may leave the king in check."""
        sign = player_sign(player)
        forward = -1 if player == WHITE else 1
        last_row = 0 if player == WHITE else SIZE - 1
        moves: List[Move] = []

        for r in range(SIZE):
            for c in range(SIZE):
                cell = self.grid[r][c]
                if cell * sign <= 0:
                    continue
                kind = abs(cell)
                from_sq = r * SIZE + c

                if kind == PAWN:
                    tr = r + forward
                    if not 0 <= tr < SIZE:
                        continue
                    targets = []
                    if self.grid[tr][c] == EMPTY:
                        targets.append(c)
                    for tc in (c - 1, c + 1):
                        if 0 <= tc < SIZE and self.grid[tr][tc] * sign < 0:
                            targets.append(tc)
                    for tc in targets:
                        to_sq = tr * SIZE + tc
                        if tr == last_row:
                            moves.extend(Move(from_sq, to_sq, p) for p in PROMOTION_PIECES)
                        else:
                            moves.append(Move(from_sq, to_sq))

                elif kind in (KNIGHT, KING):
                    steps = KNIGHT_STEPS if kind == KNIGHT else KING_STEPS
                    for dr, dc in steps:
                        tr, tc = r + dr, c + dc
                        if 0 <= tr < SIZE and 0 <= tc < SIZE and self.grid[tr][tc] * sign <= 0:
                            moves.append(Move(from_sq, tr * SIZE + tc))

                else:
                    for dr, dc in SLIDER_DIRS[kind]:
                        tr, tc = r + dr, c + dc
                        while 0 <= tr < SIZE and 0 <= tc < SIZE:
                            target = self.grid[tr][tc]
                            if target * sign > 0:
                                break
                            moves.append(Move(from_sq, tr * SIZE + tc))
                            if target != EMPTY:
                                break
                            tr += dr
                            tc += dc
        return moves

    def _is_attacked(self, sq: int, by: Player) -> bool:
        r, c = divmod(sq, SIZE)
        sign = player_sign(by)

        # A White pawn attacks diagonally upward, so it sits one row below.
        pr = r + 1 if by == WHITE else r - 1
        if 0 <= pr < SIZE:
            for pc in (c - 1, c + 1):
                if 0 <= pc < SIZE and self.grid[pr][pc] == PAWN * sign:
                    return True

        for steps, kind in ((KNIGHT_STEPS, KNIGHT), (KING_STEPS, KING)):
            for dr, dc in steps:
                tr, tc = r + dr, c + dc
                if 0 <= tr < SIZE and 0 <= tc < SIZE and self.grid[tr][tc] == kind * sign:
                    return True

        for dirs, kind in ((ROOK_DIRS, ROOK), (BISHOP_DIRS, BISHOP)):
            for dr, dc in dirs:
                tr, tc = r + dr, c + dc
                while 0 <= tr < SIZE and 0 <= tc < SIZE:
                    target = self.grid[tr][tc]
                    if target != EMPTY:
                        if target in (kind * sign, QUEEN * sign):
                            return True
                        break
                    tr += dr
                    tc += dc
        return False

    def _find_king(self, player: Player) -> Optional[int]:
        king = KING * player_sign(player)
        for r in range(SIZE):
            for c in range(SIZE):
                if self.grid[r][c] == king:
                    return r * SIZE + c
        return None

    def _insufficient_material(self) -> bool:
        """Only kings left, or kings plus a single knight or bishop."""
        minors = 0
        for row in self.grid:
            for cell in row:
                kind = abs(cell)
                if kind in (PAWN, ROOK, QUEEN):
                    return False
                if kind in (KNIGHT, BISHOP):
                    minors += 1
        return minors <= 1

    def __str__(self) -> str:
        lines = []
        for r, row in enumerate(self.grid):
            cells = []
            for cell in row:
                if cell == EMPTY:
                    cells.append(".")
                else:
                    letter = PIECE_LETTERS[abs(cell)]
                    cells.append(letter.upper() if cell > 0 else letter)
            lines.append(f"{SIZE - r} " + " ".join(cells))
        lines.append("  " + " ".join("abcde"))
        return "\n".join(lines)


def _parse_rows(rows: Sequence[str]) -> List[List[int]]:
    if len(rows) != SIZE or any(len(row) != SIZE for row in rows):
        raise ValueError(f"need {SIZE} rows of {SIZE} characters")
    grid = []
    for row in rows:
        cells = []
        for ch in row:
            if ch == ".":
                cells.append(EMPTY)
            elif ch.lower() in LETTER_PIECES:
                kind = LETTER_PIECES[ch.lower()]
                cells.append(kind if ch.isupper() else -kind)
            else:
                raise ValueError(f"bad piece letter: {ch!r}")
        grid.append(cells)
    return grid
