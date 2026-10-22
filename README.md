# Chess AI (5×5 Minichess)

A chess engine for **Gardner's minichess**, the standard 5×5 chess variant.
It has a full rules engine, a minimax AI and a self-play pipeline that records games for training a learned evaluator later.
It follows the same design as my [Connect 4 AI](https://github.com/Cookie-1412/connect4-ai).

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![No dependencies](https://img.shields.io/badge/engine-zero%20dependencies-success)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)

```
  a b c d e
5 r n b q k
4 p p p p p
3 . . . . .
2 P P P P P
1 R N B Q K
```

## Rules

- Each side starts with R N B Q K on the back rank and five pawns.
- Pieces move like in normal chess.
- Pawns only move one square so there is no en passant. There is no castling either.
- Pawns promote to a queen, rook, bishop or knight.
- Checkmate wins. Stalemate, threefold repetition, the fifty-move rule and insufficient material are draws.

## Quick start

The game itself needs only Python. NumPy and pytest are for data export and tests.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

python main.py play                          # human vs human
python main.py play --vs minimax             # you (White) vs the AI
python main.py play --vs minimax --color black
```

## Self-play data

```bash
python main.py selfplay --white minimax --black minimax -n 1000 --depth 3
python stats.py           # results summary
python dedupe.py data/    # drop repeated games
python main.py export-npz # pack into NumPy tensors
```

Games run in parallel on every CPU core but one. Each game opens with random moves so the deterministic minimax doesn't replay the same game.
`benchmark_depth.py` shows how long one move takes at each depth before you pick `--depth`.

## Project layout

```
chess5/
  board.py       pure rules engine: move generation, check, draws
  heuristics.py  material-count evaluation
  players.py     human, random and minimax players
  recorder.py    game logging to JSONL / NumPy
  game.py        game loop
main.py          CLI: play / selfplay / export-npz
tests/           pytest suite
```

## Roadmap

- [x] Rules engine and tests
- [x] Minimax player
- [x] Parallel self-play recorder
- [ ] Train a value network on self-play games (as in Connect 4 AI)
- [ ] Alpha-beta pruning for deeper search
