
import os, time, sys
from contextlib import contextmanager
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Callable

from case_closed_game import Game, Direction, GameResult, EMPTY, GameBoard, Agent

# ------------------- Server boilerplate -------------------
app = Flask(__name__)
PARTICIPANT = os.getenv("PARTICIPANT", "ParticipantX")
AGENT_NAME = os.getenv("AGENT_NAME", "MinimaxPureSafe")

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

# ------------------- Config -------------------
MAX_DEPTH = int(os.getenv("DEPTH", "10"))          # high ceiling; IDDFS will stop earlier if needed
NODE_BUDGET = int(os.getenv("NODE_BUDGET", "200000"))
TIME_BUDGET_MS = int(os.getenv("TIME_BUDGET_MS", "3600"))
ENABLE_BOOST = os.getenv("ENABLE_BOOST", "1") != "0"
BOOST_ALLOW_AT = int(os.getenv("BOOST_ALLOW_AT", "999"))  # effectively "never restrict us" for pure; change if desired

QUIET = os.getenv("QUIET", "1") != "0"

DIRS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
OPPOSITE = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT}

# Suppress stdout during internal simulations to avoid future-state prints
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

# ------------------- Helpers -------------------
def head_pos(agent: Agent) -> Tuple[int, int]:
    return agent.trail[-1]

def neighbor(board: GameBoard, pos: Tuple[int, int], d: Direction) -> Tuple[int, int]:
    x, y = pos
    dx, dy = d.value
    return ((x + dx) % board.width, (y + dy) % board.height)

def relative_dirs(cur: Direction) -> List[Direction]:
    if cur == Direction.UP:
        return [Direction.UP, Direction.LEFT, Direction.RIGHT]
    if cur == Direction.DOWN:
        return [Direction.DOWN, Direction.RIGHT, Direction.LEFT]
    if cur == Direction.LEFT:
        return [Direction.LEFT, Direction.DOWN, Direction.UP]
    # cur == Direction.RIGHT
    return [Direction.RIGHT, Direction.UP, Direction.DOWN]

def safe_ahead(board: GameBoard, pos: Tuple[int,int], d: Direction) -> bool:
    nx, ny = neighbor(board, pos, d)
    return board.grid[ny][nx] == EMPTY

def candidate_moves(agent: Agent, board: GameBoard, allow_boost: Callable[[Agent], bool]) -> List[Tuple[Direction,bool]]:
    """Return (direction, use_boost). No reverse. Boost only when policy allows & 2 cells are clear."""
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

# ------------------- Pure terminal evaluation -------------------
def terminal_value(result: GameResult, me: int, ply: int) -> float:
    if result == GameResult.DRAW: return 0.0
    if (result == GameResult.AGENT1_WIN and me == 1) or (result == GameResult.AGENT2_WIN and me == 2):
        return 1_000_000.0 - ply
    else:
        return -1_000_000.0 + ply

# ------------------- Transposition table -------------------
@dataclass(frozen=True)
class TTKey:
    rows: Tuple[str, ...]
    a1: Tuple[int,int]; a2: Tuple[int,int]
    d1: int; d2: int
    b1: int; b2: int
    al1: bool; al2: bool

@dataclass
class TTEntry:
    depth: int
    value: float
    alpha: float
    beta: float

def make_key(g: Game) -> TTKey:
    rows = tuple(''.join(str(c) for c in row) for row in g.board.grid)
    a1 = g.agent1; a2 = g.agent2
    d1 = (a1.direction.value[0] + 1) * 5 + (a1.direction.value[1] + 1)
    d2 = (a2.direction.value[0] + 1) * 5 + (a2.direction.value[1] + 1)
    return TTKey(rows, head_pos(a1), head_pos(a2), d1, d2, a1.boosts_remaining, a2.boosts_remaining, a1.alive, a2.alive)

# ------------------- Search ctx -------------------
@dataclass
class Ctx:
    me: int
    start: float
    time_budget: float
    node_budget: int
    nodes: int = 0
    tt: Dict[TTKey, TTEntry] = field(default_factory=dict)

def over_budget(ctx: Ctx) -> bool:
    if ctx.nodes >= ctx.node_budget: return True
    return (time.time() - ctx.start) >= ctx.time_budget

# ------------------- Quiescence-like extension (no heuristics) -------------------
def need_extension(g: Game) -> bool:
    """Extend search in tactically volatile states without using heuristic scoring.
    Triggers when: heads are 1-step apart OR either side has <=1 safe non-reverse move.
    """
    a1, a2 = g.agent1, g.agent2
    h1, h2 = head_pos(a1), head_pos(a2)
    W, H = g.board.width, g.board.height

    # torus manhattan distance
    dx = min((h1[0]-h2[0])%W, (h2[0]-h1[0])%W)
    dy = min((h1[1]-h2[1])%H, (h2[1]-h1[1])%H)
    if dx + dy <= 1:  # potential collision or squeeze
        return True

    def safe_count(agent: Agent):
        c = 0
        for d in DIRS:
            if OPPOSITE[d] == agent.direction: 
                continue
            if safe_ahead(g.board, head_pos(agent), d):
                c += 1
        return c
    if safe_count(a1) <= 1 or safe_count(a2) <= 1:
        return True
    return False

# ------------------- Poison-move filter -------------------
def is_poison_move(game: Game, me_id: int, move: Tuple[Direction,bool]) -> bool:
    """Returns True if for this move, there exists an opponent reply such that
    the result is an immediate loss, or all continuations at depth 1 lead to terminal loss (mate-in-1).
    This uses only terminal outcomes, keeping the 'no weights' contract.
    """
    g2 = clone_game(game)
    me = g2.agent1 if me_id == 1 else g2.agent2
    op = g2.agent2 if me_id == 1 else g2.agent1

    # Opponent moves: boost only if they have >0
    op_actions = candidate_moves(op, g2.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining > 0)
    if not op_actions:
        return False  # opponent has no legal reply => not poison

    for d_op, b_op in op_actions:
        g3 = clone_game(g2)
        try:
            with suppress_stdout():
                res = g3.step(move[0] if me_id == 1 else d_op,
                              d_op if me_id == 1 else move[0],
                              move[1] if me_id == 1 else b_op,
                              b_op if me_id == 1 else move[1])
        except Exception:
            # illegal => ignore this reply
            continue
        if res is not None:
            # if the terminal favors opponent => poison
            if (res == GameResult.AGENT1_WIN and me_id == 2) or (res == GameResult.AGENT2_WIN and me_id == 1):
                return True
        else:
            # check if after their reply, we have *no* reply avoiding terminal loss
            me2 = g3.agent1 if me_id == 1 else g3.agent2
            my_actions = candidate_moves(me2, g3.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining > 0)
            safe_exists = False
            for d2, b2 in my_actions:
                g4 = clone_game(g3)
                try:
                    with suppress_stdout():
                        res2 = g4.step(d2 if me_id == 1 else d_op,  # here d_op is stale; but res2 only checks terminal
                                       d_op if me_id == 1 else d2,
                                       b2 if me_id == 1 else b_op,
                                       b_op if me_id == 1 else b2)
                except Exception:
                    continue
                if res2 is None:
                    safe_exists = True; break
                # if terminal, see if it is not a loss for us
                if (res2 == GameResult.DRAW) or                    (res2 == GameResult.AGENT1_WIN and me_id == 1) or                    (res2 == GameResult.AGENT2_WIN and me_id == 2):
                    safe_exists = True; break
            if not safe_exists:
                return True
    return False

# ------------------- Move ordering (non-weighted) -------------------
def order_moves(game: Game, me: Agent, moves: List[Tuple[Direction,bool]]) -> List[Tuple[Direction,bool]]:
    """Deterministic, non-weighted preferences:
       1) safe steps before unsafe
       2) non-boost before boost (to control branching)
       3) tie-breaker: keep current direction if safe, else fixed dir order
    """
    cur = me.direction
    def key(m):
        d,b = m
        safe = safe_ahead(game.board, head_pos(me), d)
        keep = (d == cur)
        return (0 if safe else 1, 0 if not b else 1, 0 if keep else 1, d.value)
    return sorted(moves, key=key)

# ------------------- Minimax + alpha-beta + IDDFS -------------------
def minimax(game: Game, depth: int, ply: int, alpha: float, beta: float, ctx: Ctx) -> float:
    ctx.nodes += 1
    if depth == 0:
        # quiescence-like extension
        if need_extension(game):
            depth += 1
        else:
            return 0.0

    if over_budget(ctx):
        return 0.0

    key = make_key(game)
    tte = ctx.tt.get(key)
    if tte is not None and tte.depth >= depth and tte.alpha <= alpha and tte.beta >= beta:
        return tte.value

    me_agent = game.agent1 if ctx.me == 1 else game.agent2
    op_agent = game.agent2 if ctx.me == 1 else game.agent1

    my_actions = candidate_moves(me_agent, game.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining <= BOOST_ALLOW_AT)
    op_actions = candidate_moves(op_agent, game.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining > 0)

    # poison filter: drop moves that are immediate traps if any non-poison exists
    non_poison = [m for m in my_actions if not is_poison_move(game, ctx.me, m)]
    if non_poison:
        my_actions = non_poison

    my_actions = order_moves(game, me_agent, my_actions)

    best = -1e18
    for d_me, b_me in my_actions:
        worst = 1e18
        for d_op, b_op in op_actions:
            g2 = clone_game(game)
            try:
                with suppress_stdout():
                    res = g2.step(d_me if ctx.me == 1 else d_op,
                                  d_op if ctx.me == 1 else d_me,
                                  b_me if ctx.me == 1 else b_op,
                                  b_op if ctx.me == 1 else b_me)
            except Exception:
                continue

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
    ctx = Ctx(me=player_number, start=time.time(), time_budget=TIME_BUDGET_MS/1000.0, node_budget=NODE_BUDGET)
    me = game.agent1 if player_number == 1 else game.agent2

    # Fallback: first safe non-reverse; else keep direction
    fallback = None
    for d,_ in candidate_moves(me, game.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining <= BOOST_ALLOW_AT):
        if d == OPPOSITE[me.direction]: 
            continue
        if safe_ahead(game.board, head_pos(me), d):
            fallback = (d, False); break
    if fallback is None:
        fallback = (me.direction, False)

    best_action = fallback
    # Prepare the 6 action set: straight, left, right and their boosted variants (no reverse)
    base_dirs = relative_dirs(me.direction)
    six_actions = []
    for d in base_dirs:
        six_actions.append((d, False))
        six_actions.append((d, True))
    # Map to hold final weights per action string key
    weights = { (f"{d.name}:BOOST" if b else d.name): float('-1e18') for d,b in six_actions }
    alpha0, beta0 = -1e18, 1e18

    # IDDFS loop
    for depth in range(1, MAX_DEPTH+1):
        local_best = None
        local_best_val = -1e18

        # Evaluate exactly the six actions (reverse excluded by construction)
        me_actions = six_actions

        for d_me, b_me in me_actions:
            g2 = clone_game(game)
            op2 = g2.agent2 if player_number == 1 else g2.agent1
            op_actions = candidate_moves(op2, g2.board, allow_boost=lambda a: ENABLE_BOOST and a.boosts_remaining > 0)

            worst = 1e18; alpha, beta = alpha0, beta0
            for d_op, b_op in op_actions:
                g3 = clone_game(g2)
                try:
                    with suppress_stdout():
                        res = g3.step(d_me if player_number == 1 else d_op,
                                      d_op if player_number == 1 else d_me,
                                      b_me if player_number == 1 else b_op,
                                      b_op if player_number == 1 else b_me)
                except Exception:
                    continue
                if res is not None:
                    val = terminal_value(res, player_number, ply=1)
                else:
                    val = minimax(g3, depth-1, ply=2, alpha=alpha, beta=beta, ctx=ctx)

                if val < worst: worst = val
                if worst <= alpha: break
                if over_budget(ctx): break

            if worst > local_best_val:
                local_best_val = worst; local_best = (d_me, b_me)
            # Update weight for this action after this depth
            key = f"{d_me.name}:BOOST" if b_me else d_me.name
            weights[key] = worst

            if over_budget(ctx): break

        if local_best is not None:
            best_action = local_best
        if over_budget(ctx): break

    # Print weights for each move (only weights; no future states)
    # Ensure consistent ordering: Straight, Straight:BOOST, Left, Left:BOOST, Right, Right:BOOST
    straight, left, right = base_dirs[0], base_dirs[1], base_dirs[2]
    ordered_keys = [
        straight.name, f"{straight.name}:BOOST",
        left.name,     f"{left.name}:BOOST",
        right.name,    f"{right.name}:BOOST",
    ]
    ordered_weights = {k: float(weights.get(k, float('-1e18'))) for k in ordered_keys}
    print(f"Weights: {ordered_weights}")

    d, use_boost = best_action
    return f"{d.name}:BOOST" if use_boost else d.name

# ------------------- HTTP -------------------
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
