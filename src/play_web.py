#!/usr/bin/env python3
"""
Visual chess UI to play against the neural network.

Usage: python3 play_web.py [model_path] [mcts_steps]
Opens a browser at http://localhost:8080
"""

import sys
import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import webbrowser
from AlphaZero_player import AlphaZero
from chess_board import board as c_board
import config

model_path = sys.argv[1] if len(sys.argv) > 1 else f"{config.rootDir}/data/model_data/latest.gz"
mcts_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50

if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    sys.exit(1)

print(f"Loading model: {model_path}")
print(f"MCTS steps: {mcts_steps}")
ai = AlphaZero(model_path, mcts_steps)

def get_legal_moves_uci(board_obj):
    actions = board_obj.actions()
    uci_moves = []
    for action in actions:
        (iy, ix), (fy, fx), prom = action
        move = chr(int(ix) + 97) + str(8 - int(iy)) + chr(int(fx) + 97) + str(8 - int(fy))
        if prom is not None:
            move += str(prom)[:1].lower()
        uci_moves.append(move)
    return uci_moves

def replay_moves(moves):
    board = c_board()
    for move in moves:
        board = ai.doMove(board, move)
    return board

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html_path = os.path.join(os.path.dirname(__file__), 'play_web.html')
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())
            return

        if parsed.path == '/api/move':
            params = parse_qs(parsed.query)
            moves_str = params.get('moves', [''])[0]
            moves = moves_str.split(',') if moves_str else []

            try:
                ai_move = ai.getMove(moves)
                if ai_move is None or ai_move == "onlykings":
                    resp = {'status': 'gameover', 'reason': 'no_moves'}
                else:
                    moves.append(ai_move)
                    board = replay_moves(moves)
                    legal = get_legal_moves_uci(board)
                    resp = {'status': 'ok', 'move': ai_move, 'legal': legal}
            except Exception as e:
                resp = {'status': 'error', 'message': str(e)}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
            return

        if parsed.path == '/api/legal':
            params = parse_qs(parsed.query)
            moves_str = params.get('moves', [''])[0]
            moves = moves_str.split(',') if moves_str else []

            try:
                board = replay_moves(moves)
                legal = get_legal_moves_uci(board)
                resp = {'status': 'ok', 'legal': legal}
            except Exception as e:
                resp = {'status': 'error', 'message': str(e)}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs

port = 8080
print(f"Starting server at http://localhost:{port}")
webbrowser.open(f"http://localhost:{port}")
HTTPServer(('', port), Handler).serve_forever()
