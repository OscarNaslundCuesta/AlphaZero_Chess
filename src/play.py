#!/usr/bin/env python3
"""
Play against the neural network in the terminal.

Usage: python3 play.py [model_path] [mcts_steps]
  model_path  — path to .gz model (default: latest.gz)
  mcts_steps  — MCTS search steps (default: 50, higher = stronger but slower)

Example:
  python3 play.py
  python3 play.py /Users/oscarnc/chess/data/model_data/latest.gz 100

Enter moves in UCI format: e2e4, g1f3, e7e8q (promotion)
Type 'quit' to exit, 'board' to reprint the board.
"""

import sys
import os
from AlphaZero_player import AlphaZero
from chess_board import board as c_board
import config

PIECE_SYMBOLS = {
    'K': '\u2654', 'Q': '\u2655', 'R': '\u2656', 'B': '\u2657', 'N': '\u2658', 'P': '\u2659',
    'k': '\u265A', 'q': '\u265B', 'r': '\u265C', 'b': '\u265D', 'n': '\u265E', 'p': '\u265F',
    ' ': ' '
}

def print_board(board_obj):
    print()
    print("    a   b   c   d   e   f   g   h")
    print("  +---+---+---+---+---+---+---+---+")
    for row in range(8):
        rank = str(8 - row)
        pieces = []
        for col in range(8):
            p = board_obj.current_board[row, col]
            pieces.append(f" {PIECE_SYMBOLS.get(p, p)} ")
        print(f"{rank} |{'|'.join(pieces)}|")
        print("  +---+---+---+---+---+---+---+---+")
    print("    a   b   c   d   e   f   g   h")
    print()

def get_legal_moves_uci(board_obj, alpha_player):
    """Get legal moves as UCI strings."""
    actions = board_obj.actions()
    uci_moves = []
    for action in actions:
        (iy, ix), (fy, fx), prom = action
        move = chr(int(ix) + 97) + str(8 - int(iy)) + chr(int(fx) + 97) + str(8 - int(fy))
        if prom is not None:
            move += str(prom)[:1].lower()
        uci_moves.append(move)
    return uci_moves

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else f"{config.rootDir}/data/model_data/latest.gz"
    mcts_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    print(f"MCTS steps: {mcts_steps}")
    ai = AlphaZero(model_path, mcts_steps)

    moves = []
    board = c_board()

    color = input("Play as white or black? (w/b): ").strip().lower()
    human_is_white = color != 'b'

    if human_is_white:
        print("You are White. AI is Black.")
    else:
        print("You are Black. AI is White.")

    print_board(board)

    while True:
        if (len(moves) % 2 == 0) == human_is_white:
            # Human's turn
            legal = get_legal_moves_uci(board, ai)
            if not legal:
                if board.check_status():
                    print("Checkmate! AI wins.")
                else:
                    print("Stalemate! Draw.")
                break

            while True:
                move = input(f"Your move ({len(moves)//2 + 1}): ").strip().lower()
                if move == 'quit':
                    print("Goodbye.")
                    sys.exit(0)
                if move == 'board':
                    print_board(board)
                    continue
                if move == 'moves':
                    print(f"Legal moves: {' '.join(sorted(legal))}")
                    continue
                if move in legal:
                    break
                print(f"Illegal move. Type 'moves' to see legal moves.")

            moves.append(move)
            board = ai.doMove(board, move)
            print_board(board)
        else:
            # AI's turn
            print("AI is thinking...")
            ai_move = ai.getMove(moves)
            if ai_move is None or ai_move == "onlykings":
                print("Game over — draw or no legal moves.")
                break
            print(f"AI plays: {ai_move}")
            moves.append(ai_move)
            board = ai.doMove(board, ai_move)
            print_board(board)
