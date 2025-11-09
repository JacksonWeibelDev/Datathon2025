
import os
import time
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

from case_closed_game import Game, Direction, GameResult, EMPTY, GameBoard, Agent

app = Flask(__name__)

PARTICIPANT = os.getenv("PARTICIPANT", "ParticipantX")
AGENT_NAME = os.getenv("AGENT_NAME", "BookedMinimaxHorizontalCutoffV4")

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

# ------------ Config ------------
MAX_DEPTH = int(os.getenv("DEPTH", "4"))
NODE_BUDGET = int(os.getenv("NODE_BUDGET", "120000"))
TIME_BUDGET_MS = int(os.getenv("TIME_BUDGET_MS", "180"))
OPENING_TURNS = int(os.getenv("OPENING_TURNS", "18"))  # scripted opening horizon
BOOST_ALLOW_AT = int(os.getenv("BOOST_ALLOW_AT", "2"))  # do not boost until boosts_remaining <= this value
ENABLE_BOOST = os.getenv("ENABLE_BOOST", "1") != "0"
QUIET = os.getenv("QUIET", "1") != "0"  # if True, suppress prints

# Tactical thresholds for the "partition override"
CUT_MIN_MARGIN = int(os.getenv("CUT_MIN_MARGIN", "6"))   # my_space - op_space >= this => force cut move
CUT_MAX_OP     = int(os.getenv("CUT_MAX_OP", "20"))      # or opponent space <= this => force cut move

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

def candidate_moves(agent: Agent, board: GameBoard, allow_boost: Callable[[Agent], bool]):
    # Non-reversing, optional safe 2-cell boost policy
    try:
        moves = []
        for d in DIRS:
            if OPPOSITE[d] == agent.direction:
                continue
            is_safe = safe_ahead(board, head_pos(agent), d)
            moves.append((d, False))
            if ENABLE_BOOST and allow_boost(agent) and is_safe:
                nx, ny = neighbor(board, head_pos(agent), d)
                nx2, ny2 = neighbor(board, (nx, ny), d)
                if board.grid[ny2][nx2] == EMPTY:
                    moves.append((d, True))
        moves = [(d,b) for (d,b) in moves if d != OPPOSITE[agent.direction]]
        return moves
    except Exception:
        return []

# --------- Clone & Eval helpers ---------
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

def evaluate(game: Game, me: int) -> float:
    a_me = game.agent1 if me == 1 else game.agent2
    a_op = game.agent2 if me == 1 else game.agent1

    if not a_me.alive and not a_op.alive: return 0.0
    if not a_me.alive: return -1e6
    if not a_op.alive: return 1e6

    my_head = head_pos(a_me); op_head = head_pos(a_op)

    # seed next
    my_next = None
    for d,_ in candidate_moves(a_me, game.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining <= BOOST_ALLOW_AT):
        if safe_ahead(game.board, my_head, d):
            my_next = neighbor(game.board, my_head, d); break
    if my_next is None: my_next = my_head

    op_next = None
    for d,_ in candidate_moves(a_op, game.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining > 0):
        if safe_ahead(game.board, op_head, d):
            op_next = neighbor(game.board, op_head, d); break
    if op_next is None: op_next = op_head

    my_space = flood_count(game.board, my_next)
    op_space = flood_count(game.board, op_next)
    return float(my_space - 2.0 * op_space)

# --------- Partition override probe ---------
def min_op_space_after_move(game: Game, me_id: int, move: Tuple[Direction,bool]) -> Tuple[int,int]:
    """Returns (worst_op_space, best_margin) over opponent replies after we play `move`.
       worst_op_space = min over opponent replies of their local flood space
       best_margin    = min over replies of (my_space - op_space)
    """
    g2 = clone_game(game)
    me = g2.agent1 if me_id == 1 else g2.agent2
    op = g2.agent2 if me_id == 1 else g2.agent1

    # Apply our move vs each opponent reply
    worst_op = 10**9
    best_margin = -10**9

    # Opponent: allow boost only if >0
    op_actions = candidate_moves(op, g2.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining > 0)

    for d_op, b_op in op_actions:
        g3 = clone_game(g2)
        try:
            res = g3.step(move[0] if me_id == 1 else d_op,
                          d_op if me_id == 1 else move[0],
                          move[1] if me_id == 1 else b_op,
                          b_op if me_id == 1 else move[1])
        except Exception:
            continue
        # evaluate local spaces from the resulting next steps
        a_me = g3.agent1 if me_id == 1 else g3.agent2
        a_op = g3.agent2 if me_id == 1 else g3.agent1
        my_h = head_pos(a_me); op_h = head_pos(a_op)

        # next seeds
        my_next = my_h
        for d,_ in candidate_moves(a_me, g3.board, allow_boost=lambda a: False):
            if safe_ahead(g3.board, my_h, d):
                my_next = neighbor(g3.board, my_h, d); break
        op_next = op_h
        for d,_ in candidate_moves(a_op, g3.board, allow_boost=lambda a: False):
            if safe_ahead(g3.board, op_h, d):
                op_next = neighbor(g3.board, op_h, d); break

        my_space = flood_count(g3.board, my_next)
        op_space = flood_count(g3.board, op_next)

        worst_op = min(worst_op, op_space)
        best_margin = max(best_margin, my_space - op_space)

    if worst_op == 10**9:
        worst_op = 0
    if best_margin == -10**9:
        best_margin = -10**6
    return worst_op, best_margin

def partition_override(game: Game, me_id: int) -> Optional[Tuple[Direction,bool]]:
    """If a horizontal move produces a clearly winning partition (by flood metrics),
       choose it immediately and skip deeper search.
    """
    me = game.agent1 if me_id == 1 else game.agent2
    hx, hy = head_pos(me)
    # only consider horizontals
    cands = []
    for d in (Direction.LEFT, Direction.RIGHT):
        if d == OPPOSITE[me.direction]: 
            continue
        if safe_ahead(game.board, (hx,hy), d):
            # our boosts must obey threshold; but for partition, prefer no-boost to keep simple
            cands.append((d, False))
    if not cands:
        return None

    best = None
    best_key = None
    for mv in cands:
        worst_op, best_margin = min_op_space_after_move(game, me_id, mv)
        # decide if this is a strong partition cut
        if worst_op <= CUT_MAX_OP or best_margin >= CUT_MIN_MARGIN:
            # pick the strongest by (op_space asc, margin desc)
            key = (worst_op, -best_margin)
            if best is None or key < best_key:
                best = mv; best_key = key
    return best

# --------- Minimax with alpha-beta ---------
@dataclass
class SearchContext:
    me: int
    node_count: int = 0
    start_time: float = 0.0
    time_budget: float = 0.18

def over_budget(ctx: SearchContext) -> bool:
    if ctx.node_count >= NODE_BUDGET: return True
    if (time.time() - ctx.start_time) >= ctx.time_budget: return True
    return False

def minimax(game: Game, depth: int, alpha: float, beta: float, ctx: SearchContext) -> float:
    try:
        ctx.node_count += 1
        if depth == 0 or over_budget(ctx):
            return evaluate(game, ctx.me)

        me_agent = game.agent1 if ctx.me == 1 else game.agent2
        op_agent = game.agent2 if ctx.me == 1 else game.agent1

        my_actions = candidate_moves(me_agent, game.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining <= BOOST_ALLOW_AT)
        op_actions = candidate_moves(op_agent, game.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining > 0)

        # Order: prefer safe horizontals, boosts last
        my_h = head_pos(me_agent)
        def order_key(ab):
            d,b = ab
            is_horizontal = d in (Direction.LEFT, Direction.RIGHT)
            safety = 0 if safe_ahead(game.board, my_h, d) else 1
            boost_bias = 1 if b else 0
            return (safety, 0 if is_horizontal else 1, boost_bias)
        my_actions.sort(key=order_key)

        best = -1e18
        for d_me, b_me in my_actions:
            worst = 1e18
            for d_op, b_op in op_actions:
                g2 = clone_game(game)
                try:
                    res = g2.step(d_me if ctx.me == 1 else d_op,
                                  d_op if ctx.me == 1 else d_me,
                                  b_me if ctx.me == 1 else b_op,
                                  b_op if ctx.me == 1 else b_me)
                except Exception:
                    continue
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
    except Exception:
        return 0.0

def choose_move(game: Game, player_number: int) -> str:
    me = game.agent1 if player_number == 1 else game.agent2

    # Tactical partition override FIRST
    mv = partition_override(game, player_number)
    if mv is not None:
        d, b = mv
        return f"{d.name}:BOOST" if (b and ENABLE_BOOST) else d.name

    # Opening (light) — still attempts horizontal slice in early turns
    mv = opening_move(game, player_number)
    if mv is not None:
        d, _ = mv
        return d.name

    # Minimax fallback
    ctx = SearchContext(me=player_number, start_time=time.time(), time_budget=TIME_BUDGET_MS/1000.0)

    # Safe fallback
    fallback = None
    for d,_ in candidate_moves(me, game.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining <= BOOST_ALLOW_AT):
        if d == OPPOSITE[me.direction]: continue
        if safe_ahead(game.board, head_pos(me), d):
            fallback = (d, False); break
    if fallback is None:
        fallback = (me.direction, False)

    best_action = fallback
    for depth in range(1, MAX_DEPTH+1):
        local_best = None
        local_best_val = -1e18

        me_actions = candidate_moves(me, game.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining <= BOOST_ALLOW_AT)
        me_actions = [ab for ab in me_actions if ab[0] != OPPOSITE[me.direction]]

        # order again (horizontals first)
        my_h = head_pos(me)
        def order_key(ab):
            d,b = ab
            is_horizontal = d in (Direction.LEFT, Direction.RIGHT)
            safety = 0 if safe_ahead(game.board, my_h, d) else 1
            boost_bias = 1 if b else 0
            return (safety, 0 if is_horizontal else 1, boost_bias)
        me_actions.sort(key=order_key)

        alpha = -1e18; beta = 1e18
        for d_me, b_me in me_actions:
            g2 = clone_game(game)
            op2 = g2.agent2 if player_number == 1 else g2.agent1
            op_actions = candidate_moves(op2, g2.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining > 0)

            worst = 1e18
            for d_op, b_op in op_actions:
                g3 = clone_game(g2)
                try:
                    res = g3.step(d_me if player_number == 1 else d_op,
                                  d_op if player_number == 1 else d_me,
                                  b_me if player_number == 1 else b_op,
                                  b_op if player_number == 1 else b_me)
                except Exception:
                    continue
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
                local_best_val = worst
                local_best = (d_me, b_me)

            if over_budget(ctx): break

        if local_best is not None:
            best_action = local_best
        if over_budget(ctx): break

    d, use_boost = best_action
    if use_boost and me.boosts_remaining > BOOST_ALLOW_AT:
        use_boost = False
    return f"{d.name}:BOOST" if (use_boost and ENABLE_BOOST) else d.name

# --------- Opening function (kept minimal – uses the same as previous version) ---------
def opening_move(game: Game, me_id: int) -> Optional[Tuple[Direction, bool]]:
    try:
        me = game.agent1 if me_id == 1 else game.agent2
        op = game.agent2 if me_id == 1 else game.agent1
        board = game.board

        if not (me.alive and op.alive):
            return None
        if game.turns >= OPENING_TURNS:
            return None

        hx, hy = head_pos(me)
        ox, oy = head_pos(op)
        W, H = board.width, board.height

        dy_down = (oy - hy) % H
        dy_up   = (hy - oy) % H

        if hy != oy and min(dy_down, dy_up) > 0:
            d = Direction.DOWN if dy_down <= dy_up else Direction.UP
            if d == OPPOSITE[me.direction]:
                d = me.direction
            if safe_ahead(board, (hx, hy), d):
                return (d, False)
            dx_right = (ox - hx) % W
            dx_left  = (hx - ox) % W
            dh = Direction.RIGHT if dx_right <= dx_left else Direction.LEFT
            if dh != OPPOSITE[me.direction] and safe_ahead(board, (hx, hy), dh):
                return (dh, False)
            return None

        dx_right = (ox - hx) % W
        dx_left  = (hx - ox) % W
        prefer = Direction.LEFT if dx_left >= dx_right else Direction.RIGHT
        if prefer == OPPOSITE[me.direction]:
            prefer = me.direction
        if safe_ahead(board, (hx, hy), prefer):
            return (prefer, False)
        alt = Direction.RIGHT if prefer == Direction.LEFT else Direction.LEFT
        if alt != OPPOSITE[me.direction] and safe_ahead(board, (hx, hy), alt):
            return (alt, False)
        for d in [Direction.UP, Direction.DOWN]:
            if d != OPPOSITE[me.direction] and safe_ahead(board, (hx, hy), d):
                return (d, False)
        return None
    except Exception:
        return None

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
    if not QUIET:
        print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
