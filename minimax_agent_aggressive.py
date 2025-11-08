
import os
import sys
from contextlib import contextmanager
import time
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
from dataclasses import dataclass
from typing import Tuple

from case_closed_game import Game, Direction, GameResult, EMPTY, GameBoard, Agent

app = Flask(__name__)

# Identity (configurable via env)
PARTICIPANT = os.getenv("PARTICIPANT", "ParticipantX")
AGENT_NAME = os.getenv("AGENT_NAME", "MinimaxAgentAggressive")

# Global mutable state mirrors the judge protocol
GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

# ---- Minimax configuration (env-tunable) ----
MAX_DEPTH = int(os.getenv("DEPTH", "3"))
NODE_BUDGET = int(os.getenv("NODE_BUDGET", "9000"))
TIME_BUDGET_MS = int(os.getenv("TIME_BUDGET_MS", "45"))
ENABLE_BOOST = os.getenv("ENABLE_BOOST", "1") != "0"

# Suppress stdout during internal simulations to avoid game prints
@contextmanager
def suppress_stdout():
    saved = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.stdout = saved

# Aggression knobs
VORONOI_WEIGHT = float(os.getenv("VORONOI_W", "6.0"))
SPACE_WEIGHT   = float(os.getenv("SPACE_W", "3.0"))
LENGTH_WEIGHT  = float(os.getenv("LEN_W", "0.3"))
BOOST_WEIGHT   = float(os.getenv("BOOST_W", "0.4"))
DIST_WEIGHT    = float(os.getenv("DIST_W", "-0.3"))  # negative => prefers being closer to the opponent

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
        # hard block reverse
        if OPPOSITE[d] == agent.direction:
            continue
        is_safe = safe_ahead(board, head_pos(agent), d)
        # base move (even if unsafe, in case we're forced)
        moves.append((d, False))
        # aggressive: prefer adding boost options when path is at least two cells clear
        if ENABLE_BOOST and agent.boosts_remaining > 0 and is_safe:
            nx, ny = neighbor(board, head_pos(agent), d)
            nx2, ny2 = neighbor(board, (nx, ny), d)
            if board.grid[ny2][nx2] == EMPTY:
                moves.append((d, True))
    # filter out any latent reverse by direction equality guard (double safety)
    moves = [(d,b) for (d,b) in moves if d != OPPOSITE[agent.direction]]
    return moves

# --------- State clone & evaluation ---------
def clone_game(game: Game) -> Game:
    g = Game()
    # Copy board
    g.board = GameBoard(height=game.board.height, width=game.board.width)
    g.board.grid = [row[:] for row in game.board.grid]
    # Copy agents
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

from collections import deque as dq
def flood_count(board: GameBoard, start: Tuple[int,int]) -> int:
    if start is None:
        return 0
    H, W = board.height, board.width
    seen = [[False]*W for _ in range(H)]
    q = dq()
    if board.grid[start[1]][start[0]] == EMPTY:
        q.append(start); seen[start[1]][start[0]] = True
    cnt = 0
    while q:
        x, y = q.popleft(); cnt += 1
        for d in DIRS:
            nx = (x + d.value[0]) % W; ny = (y + d.value[1]) % H
            if not seen[ny][nx] and board.grid[ny][nx] == EMPTY:
                seen[ny][nx] = True; q.append((nx, ny))
    return cnt

def manhattan_torus(a: Tuple[int,int], b: Tuple[int,int], W: int, H: int) -> int:
    dx = min((a[0]-b[0])%W, (b[0]-a[0])%W)
    dy = min((a[1]-b[1])%H, (b[1]-a[1])%H)
    return dx + dy

def voronoi_score(board: GameBoard, h1: Tuple[int,int], h2: Tuple[int,int]) -> Tuple[int,int]:
    """Multi-source BFS to count cells closer to h1 vs h2 (ties ignored)."""
    H, W = board.height, board.width
    dist1 = [[None]*W for _ in range(H)]
    dist2 = [[None]*W for _ in range(H)]
    q1, q2 = dq(), dq()
    if board.grid[h1[1]][h1[0]] == EMPTY: q1.append(h1); dist1[h1[1]][h1[0]] = 0
    if board.grid[h2[1]][h2[0]] == EMPTY: q2.append(h2); dist2[h2[1]][h2[0]] = 0
    while q1:
        x,y = q1.popleft()
        for d in DIRS:
            nx=(x+d.value[0])%W; ny=(y+d.value[1])%H
            if dist1[ny][nx] is None and board.grid[ny][nx] == EMPTY:
                dist1[ny][nx] = dist1[y][x] + 1
                q1.append((nx, ny))
    while q2:
        x,y = q2.popleft()
        for d in DIRS:
            nx=(x+d.value[0])%W; ny=(y+d.value[1])%H
            if dist2[ny][nx] is None and board.grid[ny][nx] == EMPTY:
                dist2[ny][nx] = dist2[y][x] + 1
                q2.append((nx, ny))
    v1 = v2 = 0
    for y in range(H):
        for x in range(W):
            if board.grid[y][x] != EMPTY: continue
            d1 = dist1[y][x]; d2 = dist2[y][x]
            if d1 is None and d2 is None: continue
            if d2 is None or (d1 is not None and d1 < d2): v1 += 1
            elif d1 is None or (d2 is not None and d2 < d1): v2 += 1
            # ties ignored = neutral
    return v1, v2

def evaluate(game: Game, me: int) -> float:
    a_me = game.agent1 if me == 1 else game.agent2
    a_op = game.agent2 if me == 1 else game.agent1

    if not a_me.alive and not a_op.alive: return 0.0
    if not a_me.alive: return -1e6
    if not a_op.alive: return 1e6

    my_head = head_pos(a_me); op_head = head_pos(a_op)

    # Next-step seeds for flood count
    my_next = None
    for d,_ in candidate_moves(a_me, game.board):
        if safe_ahead(game.board, my_head, d):
            my_next = neighbor(game.board, my_head, d); break
    if my_next is None: my_next = my_head

    op_next = None
    for d,_ in candidate_moves(a_op, game.board):
        if safe_ahead(game.board, op_head, d):
            op_next = neighbor(game.board, op_head, d); break
    if op_next is None: op_next = op_head

    # Space control (local flood) and aggressive global Voronoi
    my_space = flood_count(game.board, my_next)
    op_space = flood_count(game.board, op_next)
    v_my, v_op = voronoi_score(game.board, my_head, op_head)

    length_diff = a_me.length - a_op.length
    boost_diff = a_me.boosts_remaining - a_op.boosts_remaining
    dist = manhattan_torus(my_head, op_head, game.board.width, game.board.height)

    score = (
        SPACE_WEIGHT   * (my_space - op_space) +
        VORONOI_WEIGHT * (v_my - v_op) +
        LENGTH_WEIGHT  * length_diff +
        BOOST_WEIGHT   * boost_diff +
        DIST_WEIGHT    * dist
    )
    return float(score)

# --------- Minimax with alpha-beta ---------
@dataclass
class SearchContext:
    me: int
    node_count: int = 0
    start_time: float = 0.0
    time_budget: float = 0.045

def over_budget(ctx: SearchContext) -> bool:
    if ctx.node_count >= NODE_BUDGET: return True
    if (time.time() - ctx.start_time) >= ctx.time_budget: return True
    return False

def minimax(game: Game, depth: int, alpha: float, beta: float, ctx: SearchContext) -> float:
    ctx.node_count += 1
    if depth == 0 or over_budget(ctx):
        return evaluate(game, ctx.me)

    me_agent = game.agent1 if ctx.me == 1 else game.agent2
    op_agent = game.agent2 if ctx.me == 1 else game.agent1

    my_actions = candidate_moves(me_agent, game.board)
    op_actions = candidate_moves(op_agent, game.board)

    # Move ordering: prefer closer-to-opponent and safe; boost earlier
    my_h = head_pos(me_agent); op_h = head_pos(op_agent)
    def order_key(ab):
        d,b = ab
        nxt = neighbor(game.board, my_h, d)
        prox = manhattan_torus(nxt, op_h, game.board.width, game.board.height)
        safety = 0 if safe_ahead(game.board, my_h, d) else 1
        boost_bias = 0 if b else 1
        return (safety, boost_bias, prox)
    my_actions.sort(key=order_key)

    best = -1e18
    for d_me, b_me in my_actions:
        worst = 1e18
        for d_op, b_op in op_actions:
            g2 = clone_game(game)
            with suppress_stdout():
                res = g2.step(d_me if ctx.me == 1 else d_op,
                          d_op if ctx.me == 1 else d_me,
                          b_me if ctx.me == 1 else b_op,
                          b_op if ctx.me == 1 else b_me)
            if res is not None:
                if res == GameResult.DRAW: val = 0.0
                elif (res == GameResult.AGENT1_WIN and ctx.me == 1) or (res == GameResult.AGENT2_WIN and ctx.me == 2): val = 1e6
                else: val = -1e6
            else:
                val = minimax(g2, depth-1, alpha, beta, ctx)

            if val < worst: worst = val
            if worst <= alpha: break
            if over_budget(ctx): break

        if worst > best: best = worst
        if best > alpha: alpha = best
        if beta <= alpha: break
        if over_budget(ctx): break

    return best

def choose_move(game: Game, player_number: int) -> str:
    ctx = SearchContext(me=player_number, start_time=time.time(), time_budget=TIME_BUDGET_MS/1000.0)
    me = game.agent1 if player_number == 1 else game.agent2

    # Fallback: first safe move that is NOT reverse; else keep direction if not reverse (engine will block reverse anyway)
    fallback = None
    for d,_ in candidate_moves(me, game.board):
        if d == OPPOSITE[me.direction]: 
            continue
        if safe_ahead(game.board, head_pos(me), d):
            fallback = (d, False); break
    if fallback is None:
        # ensure we do not choose reverse as fallback
        fallback_dir = me.direction if me.direction != OPPOSITE[me.direction] else me.direction
        fallback = (fallback_dir, False)

    best_action = fallback
    best_val = -1e18

    for depth in range(1, MAX_DEPTH+1):
        local_best = None
        local_best_val = -1e18

        me_actions = candidate_moves(me, game.board)

        # Aggressive ordering for top ply as well
        my_h = head_pos(me)
        op = game.agent2 if player_number == 1 else game.agent1
        op_h = head_pos(op)

        def order_key(ab):
            d,b = ab
            nxt = neighbor(game.board, my_h, d)
            prox = manhattan_torus(nxt, op_h, game.board.width, game.board.height)
            safety = 0 if safe_ahead(game.board, my_h, d) else 1
            boost_bias = 0 if b else 1
            return (safety, boost_bias, prox)

        me_actions = [ab for ab in me_actions if ab[0] != OPPOSITE[me.direction]]
        me_actions.sort(key=order_key)

        for d_me, b_me in me_actions:
            g2 = clone_game(game)
            op2 = g2.agent2 if player_number == 1 else g2.agent1
            op_actions = candidate_moves(op2, g2.board)

            worst = 1e18; alpha = -1e18; beta = 1e18
            for d_op, b_op in op_actions:
                g3 = clone_game(g2)
                with suppress_stdout():
                    res = g3.step(d_me if player_number == 1 else d_op,
                              d_op if player_number == 1 else d_me,
                              b_me if player_number == 1 else b_op,
                              b_op if player_number == 1 else b_me)
                if res is not None:
                    if res == GameResult.DRAW: val = 0.0
                    elif (res == GameResult.AGENT1_WIN and player_number == 1) or (res == GameResult.AGENT2_WIN and player_number == 2): val = 1e6
                    else: val = -1e6
                else:
                    inner_ctx = SearchContext(me=player_number, start_time=ctx.start_time, time_budget=ctx.time_budget)
                    val = minimax(g3, depth-1, alpha, beta, inner_ctx)

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

# ------------- HTTP endpoints (judge protocol) -------------
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
    port = int(os.environ.get("PORT", "5008"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
