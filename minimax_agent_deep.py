
import os
import time
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Dict

from case_closed_game import Game, Direction, GameResult, EMPTY, GameBoard, Agent

app = Flask(__name__)

# Identity
PARTICIPANT = os.getenv("PARTICIPANT", "ParticipantX")
AGENT_NAME = os.getenv("AGENT_NAME", "MinimaxAgentDeep")

# Shared game state
GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

# Deep search config (tune via env)
MAX_DEPTH = int(os.getenv("DEPTH", "10"))
NODE_BUDGET = int(os.getenv("NODE_BUDGET", "120000"))
TIME_BUDGET_MS = int(os.getenv("TIME_BUDGET_MS", "180"))
ENABLE_BOOST = os.getenv("ENABLE_BOOST", "1") != "0"

DIRS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
OPPOSITE = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT}

def head_pos(agent: Agent) -> Tuple[int, int]:
    return agent.trail[-1]

def neighbor(board: GameBoard, pos: Tuple[int, int], d: Direction) -> Tuple[int, int]:
    x, y = pos
    dx, dy = d.value
    return ((x + dx) % board.width, (y + dy) % board.height)

def safe_ahead(board: GameBoard, pos: Tuple[int,int], d: Direction) -> bool:
    nx, ny = neighbor(board, pos, d)
    return board.grid[ny][nx] == EMPTY

def candidate_moves(agent: Agent, board: GameBoard):
    """Return list of (direction, use_boost_flag). Never include a reverse move."""
    moves = []
    for d in DIRS:
        if OPPOSITE[d] == agent.direction:
            continue
        is_safe = safe_ahead(board, head_pos(agent), d)
        moves.append((d, False))
        if ENABLE_BOOST and agent.boosts_remaining > 0 and is_safe:
            nx, ny = neighbor(board, head_pos(agent), d)
            nx2, ny2 = neighbor(board, (nx, ny), d)
            if board.grid[ny2][nx2] == EMPTY:
                moves.append((d, True))
    moves = [(d,b) for (d,b) in moves if d != OPPOSITE[agent.direction]]
    return moves

def clone_game(game: Game) -> Game:
    g = Game()
    g.board = GameBoard(height=game.board.height, width=game.board.width)
    g.board.grid = [row[:] for row in game.board.grid]
    g.agent1.trail = deque(game.agent1.trail)
    g.agent1.direction = game.agent1.direction
    g.agent1.alive = game.agent1.alive
    g.agent1.length = game.agent1.length
    g.agent1.boosts_remaining = game.agent1.boosts_remaining

    g.agent2.trail = deque(game.agent2.trail)
    g.agent2.direction = game.agent2.direction
    g.agent2.alive = game.agent2.alive
    g.agent2.length = game.agent2.length
    g.agent2.boosts_remaining = game.agent2.boosts_remaining

    g.turns = game.turns
    return g

# ------------ Plain evaluation (only terminals) ------------
def terminal_value(result: GameResult, me: int, ply: int) -> float:
    if result == GameResult.DRAW: return 0.0
    if (result == GameResult.AGENT1_WIN and me == 1) or (result == GameResult.AGENT2_WIN and me == 2):
        return 1_000_000.0 - ply   # prefer faster wins
    else:
        return -1_000_000.0 + ply  # prefer slower losses

# ------------- Transposition Table -------------
from dataclasses import field

@dataclass(frozen=True)
class TTKey:
    board_rows: Tuple[str, ...]
    a1_head: Tuple[int, int]
    a2_head: Tuple[int, int]
    a1_dir: int
    a2_dir: int
    a1_alive: bool
    a2_alive: bool
    a1_boosts: int
    a2_boosts: int

@dataclass
class TTEntry:
    depth: int
    value: float
    alpha: float
    beta: float

def make_key(game: Game) -> TTKey:
    rows = tuple(''.join(str(c) for c in row) for row in game.board.grid)
    a1 = game.agent1; a2 = game.agent2
    a1_dir_code = (a1.direction.value[0] + 1) * 5 + (a1.direction.value[1] + 1)
    a2_dir_code = (a2.direction.value[0] + 1) * 5 + (a2.direction.value[1] + 1)
    return TTKey(
        board_rows=rows,
        a1_head=head_pos(a1),
        a2_head=head_pos(a2),
        a1_dir=a1_dir_code,
        a2_dir=a2_dir_code,
        a1_alive=a1.alive, a2_alive=a2.alive,
        a1_boosts=a1.boosts_remaining, a2_boosts=a2.boosts_remaining
    )

# ------------- Minimax + AlphaBeta + IDDFS -------------
@dataclass
class SearchContext:
    me: int
    node_count: int = 0
    start_time: float = 0.0
    time_budget: float = 0.18
    tt: Dict[TTKey, TTEntry] = field(default_factory=dict)

def over_budget(ctx: SearchContext) -> bool:
    if ctx.node_count >= NODE_BUDGET: return True
    if (time.time() - ctx.start_time) >= ctx.time_budget: return True
    return False

def order_moves(game: Game, moves, opp_head: Tuple[int,int], my_head: Tuple[int,int]) -> list:
    # Non-heuristic ordering: safe first, boosted second, then proximity (tie-breaker only)
    W, H = game.board.width, game.board.height
    def prox(p, q):
        dx = min((p[0]-q[0])%W, (q[0]-p[0])%W)
        dy = min((p[1]-q[1])%H, (q[1]-p[1])%H)
        return dx + dy
    def key(m):
        d,b = m
        nxt = neighbor(game.board, my_head, d)
        safety = 0 if safe_ahead(game.board, my_head, d) else 1
        boost_bias = 0 if b else 1
        return (safety, boost_bias, prox(nxt, opp_head))
    return sorted(moves, key=key)

def minimax(game: Game, depth: int, ply: int, alpha: float, beta: float, ctx: SearchContext) -> float:
    ctx.node_count += 1
    if depth == 0 or over_budget(ctx):
        return 0.0  # neutral at leaves

    # Transposition
    key = make_key(game)
    tte = ctx.tt.get(key)
    if tte is not None and tte.depth >= depth and tte.alpha <= alpha and tte.beta >= beta:
        return tte.value

    me_agent = game.agent1 if ctx.me == 1 else game.agent2
    op_agent = game.agent2 if ctx.me == 1 else game.agent1

    my_actions = candidate_moves(me_agent, game.board)
    op_actions = candidate_moves(op_agent, game.board)

    my_head = head_pos(me_agent); op_head = head_pos(op_agent)
    my_actions = order_moves(game, my_actions, opp_head=op_head, my_head=my_head)

    best = -1e18
    for d_me, b_me in my_actions:
        worst = 1e18
        for d_op, b_op in op_actions:
            g2 = clone_game(game)
            res = g2.step(d_me if ctx.me == 1 else d_op,
                          d_op if ctx.me == 1 else d_me,
                          b_me if ctx.me == 1 else b_op,
                          b_op if ctx.me == 1 else b_me)
            if res is not None:
                val = terminal_value(res, ctx.me, ply)
            else:
                val = minimax(g2, depth-1, ply+1, alpha, beta, ctx)

            if val < worst: worst = val
            if worst <= alpha: break
            if over_budget(ctx): break

        if worst > best:
            best = worst
        if best > alpha: alpha = best
        if beta <= alpha: break
        if over_budget(ctx): break

    ctx.tt[key] = TTEntry(depth=depth, value=best, alpha=alpha, beta=beta)
    return best

def choose_move(game: Game, player_number: int) -> str:
    ctx = SearchContext(me=player_number, start_time=time.time(), time_budget=TIME_BUDGET_MS/1000.0)

    me = game.agent1 if player_number == 1 else game.agent2

    # Non-reverse safe fallback
    fallback = None
    for d,_ in candidate_moves(me, game.board):
        if d == OPPOSITE[me.direction]: continue
        if safe_ahead(game.board, head_pos(me), d):
            fallback = (d, False); break
    if fallback is None:
        fallback = (me.direction, False)

    best_action = fallback
    best_val = -1e18

    alpha0 = -1e18; beta0 = 1e18
    for depth in range(1, MAX_DEPTH+1):
        local_best = None
        local_best_val = -1e18

        me_actions = candidate_moves(me, game.board)
        me_actions = [ab for ab in me_actions if ab[0] != OPPOSITE[me.direction]]

        op = game.agent2 if player_number == 1 else game.agent1
        me_head = head_pos(me); op_head = head_pos(op)
        me_actions = order_moves(game, me_actions, opp_head=op_head, my_head=me_head)

        for d_me, b_me in me_actions:
            g2 = clone_game(game)
            op2 = g2.agent2 if player_number == 1 else g2.agent1
            op_actions = candidate_moves(op2, g2.board)

            worst = 1e18; alpha = alpha0; beta = beta0
            for d_op, b_op in op_actions:
                g3 = clone_game(g2)
                res = g3.step(d_me if player_number == 1 else d_op,
                              d_op if player_number == 1 else d_me,
                              b_me if player_number == 1 else b_op,
                              b_op if player_number == 1 else b_me)
                if res is not None:
                    val = terminal_value(res, player_number, ply=1)
                else:
                    val = minimax(g3, depth-1, ply=2, alpha=alpha, beta=beta, ctx=ctx)

                if val < worst: worst = val
                if worst <= alpha: break
                if over_budget(ctx): break

            if worst > local_best_val:
                local_best_val = worst; local_best = (d_me, b_me)

            if over_budget(ctx): break

        if local_best is not None:
            best_action = local_best; best_val = local_best_val
        if over_budget(ctx): break

    d, use_boost = best_action
    return f"{d.name}:BOOST" if use_boost else d.name

# ------------- HTTP endpoints -------------
@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

def _update_local_game_from_post(data: dict):
    with game_lock:
        LAST_POSTED_STATE.clear(); LAST_POSTED_STATE.update(data)

        if "board" in data:
            try: GLOBAL_GAME.board.grid = data["board"]
            except Exception: pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"])
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])

@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()
    if not data: return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200

@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)
    with game_lock:
        move = choose_move(GLOBAL_GAME, player_number)
    return jsonify({"move": move}), 200

@app.route("/end", methods=["POST"])
def end_game():
    data = request.get_json()
    if data: _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
