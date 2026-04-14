#!/usr/bin/env python3
"""
Convert Lichess parquet data to AlphaZero training format.

Usage: python3 convert_lichess.py <parquet_file> [max_positions]
Example:
  python3 convert_lichess.py ~/chess/data/lichess_0.parquet 1000000

Reads the parquet file, groups rows by FEN, picks highest-depth evaluation
per position, builds (board_state, policy_vector, value) tuples, and saves
as .gz files in data/train/.
"""

import sys
import os
import numpy as np
import pyarrow.parquet as pq
import math
from compress_pickle import dump
from chess_board import board as c_board
import encoder_decoder as ed
import config

PIECE_MAP = {
    'R': 'R', 'N': 'N', 'B': 'B', 'Q': 'Q', 'K': 'K', 'P': 'P',
    'r': 'r', 'n': 'n', 'b': 'b', 'q': 'q', 'k': 'k', 'p': 'p'
}

def fen_to_board(fen):
    """Convert a FEN string to a chess_board.board object."""
    parts = fen.split(' ')
    piece_placement = parts[0]
    active_color = parts[1] if len(parts) > 1 else 'w'
    castling = parts[2] if len(parts) > 2 else '-'
    en_passant = parts[3] if len(parts) > 3 else '-'

    board = c_board()
    # Clear the board
    board.current_board = np.zeros([8, 8]).astype(str)
    board.current_board[board.current_board == "0.0"] = " "

    # Place pieces
    row = 0
    col = 0
    for ch in piece_placement:
        if ch == '/':
            row += 1
            col = 0
        elif ch.isdigit():
            col += int(ch)
        else:
            board.current_board[row, col] = ch
            col += 1

    # Active color
    board.player = 0 if active_color == 'w' else 1

    # Castling rights — if a right is missing, mark the piece as moved
    if 'K' not in castling:
        board.R2_move_count = 1  # white kingside rook moved
    if 'Q' not in castling:
        board.R1_move_count = 1  # white queenside rook moved
    if 'K' not in castling and 'Q' not in castling:
        board.K_move_count = 1   # white king moved
    if 'k' not in castling:
        board.r2_move_count = 1  # black kingside rook moved
    if 'q' not in castling:
        board.r1_move_count = 1  # black queenside rook moved
    if 'k' not in castling and 'q' not in castling:
        board.k_move_count = 1   # black king moved

    # En passant
    if en_passant != '-':
        ep_col = ord(en_passant[0]) - 97
        board.en_passant = ep_col
    else:
        board.en_passant = -999

    # These are unknown from FEN, zero them out
    board.move_count = 0
    board.repetitions_w = 0
    board.repetitions_b = 0
    board.no_progress_count = 0

    return board


def uci_to_positions(move_str):
    """Convert UCI move string like 'e2e4' to ((from_row, from_col), (to_row, to_col), promote)."""
    from_col = ord(move_str[0]) - 97
    from_row = 8 - int(move_str[1])
    to_col = ord(move_str[2]) - 97
    to_row = 8 - int(move_str[3])
    promote = None
    if len(move_str) == 5:
        p = move_str[4].lower()
        if p == 'q':
            promote = 'queen'
        elif p == 'r':
            promote = 'rook'
        elif p == 'n':
            promote = 'knight'
        elif p == 'b':
            promote = 'bishop'
    return (from_row, from_col), (to_row, to_col), promote


def softmax(x):
    x = np.array(x, dtype=np.float64)
    x = x / 20.0  # temperature, same as generate.py
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def process_position(fen, rows):
    """Convert a FEN + its candidate move rows into a training example."""
    try:
        board = fen_to_board(fen)
    except Exception:
        return None

    # Pick highest depth per move (first move of each line)
    best_by_move = {}
    for line, depth, cp, mate in rows:
        first_move = line.split()[0] if line else None
        if not first_move:
            continue

        # Convert mate/cp to a unified score
        if mate is not None:
            score = 10000 - abs(mate) if mate > 0 else -10000 + abs(mate)
        elif cp is not None:
            score = cp
        else:
            continue

        if first_move not in best_by_move or depth > best_by_move[first_move][1]:
            best_by_move[first_move] = (score, depth, first_move)

    if not best_by_move:
        return None

    moves = list(best_by_move.values())
    scores = [m[0] for m in moves]
    move_strs = [m[2] for m in moves]

    # Value: best score from current player's perspective
    # cp is from white's perspective, negate for black
    best_score = max(scores)
    if board.player == 1:  # black to move
        best_score = -best_score
    value = math.tanh(best_score / 400.0)

    # Policy: softmax over scores (from current player's perspective)
    if board.player == 1:
        policy_scores = [-s for s in scores]
    else:
        policy_scores = scores
    probs = softmax(policy_scores)

    # Build policy vector
    policy = np.zeros(4672)
    for move_str, prob in zip(move_strs, probs):
        try:
            initial, final, promote = uci_to_positions(move_str)
            enc_index = ed.encode_action(board, initial, final, promote)
            policy[enc_index] = prob
        except Exception:
            continue

    if policy.sum() == 0:
        return None

    # Encode board
    board_state = ed.encode_board(board)

    return [board_state, policy, value]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert_lichess.py <parquet_file> [max_positions]")
        sys.exit(1)

    parquet_path = sys.argv[1]
    max_positions = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000

    out_dir = f"{config.rootDir}/data/train"
    save_every = 10000
    file_count = 0
    dataset = []
    positions_done = 0
    positions_skipped = 0

    print(f"Reading {parquet_path}...")
    pf = pq.ParquetFile(parquet_path)
    print(f"Total rows: {pf.metadata.num_rows}, row groups: {pf.metadata.num_row_groups}")

    # Process row group by row group to limit memory
    current_fen = None
    current_rows = []

    for rg_idx in range(pf.metadata.num_row_groups):
        if positions_done >= max_positions:
            break

        table = pf.read_row_group(rg_idx, columns=['fen', 'line', 'depth', 'cp', 'mate'])
        fens = table['fen'].to_pylist()
        lines = table['line'].to_pylist()
        depths = table['depth'].to_pylist()
        cps = table['cp'].to_pylist()
        mates = table['mate'].to_pylist()

        for i in range(len(fens)):
            fen = fens[i]

            if fen != current_fen:
                # Process previous position
                if current_fen is not None and current_rows:
                    result = process_position(current_fen, current_rows)
                    if result is not None:
                        dataset.append(result)
                        positions_done += 1
                    else:
                        positions_skipped += 1

                    if positions_done >= max_positions:
                        break

                    if len(dataset) >= save_every:
                        filepath = f"{out_dir}/lichess_{file_count}.gz"
                        print(f"Saving {len(dataset)} positions to {filepath} (total: {positions_done}, skipped: {positions_skipped})")
                        with open(filepath, 'wb') as f:
                            dump(dataset, f)
                        dataset = []
                        file_count += 1

                current_fen = fen
                current_rows = []

            current_rows.append((lines[i], depths[i], cps[i], mates[i]))

    # Process last position
    if current_fen and current_rows and positions_done < max_positions:
        result = process_position(current_fen, current_rows)
        if result is not None:
            dataset.append(result)
            positions_done += 1

    # Save remaining
    if dataset:
        filepath = f"{out_dir}/lichess_{file_count}.gz"
        print(f"Saving {len(dataset)} positions to {filepath} (total: {positions_done}, skipped: {positions_skipped})")
        with open(filepath, 'wb') as f:
            dump(dataset, f)

    print(f"Done. Converted {positions_done} positions, skipped {positions_skipped}.")
