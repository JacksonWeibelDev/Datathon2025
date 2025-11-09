import os, time, random
from collections import deque
from threading import Lock
from flask import Flask, request, jsonify
from case_closed_game import Game, Direction

# =========================
# Time budget (stay < 4 s)
# =========================
MOVE_BUDGET_SEC = 3.5  # keep margin for HTTP

class Deadline:
    __slots__ = ("t_end",)
    def __init__(self, seconds): self.t_end = time.perf_counter() + seconds
    def time_left(self): return self.t_end - time.perf_counter()
    def expired(self):   return time.perf_counter() >= self.t_end

# =========================
# Flask / global state
# =========================
app = Flask(__name__)
game_lock = Lock()
GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

AGENT_NAME = os.environ.get("AGENT_NAME", "Weighted6Ply")
AGENT_ID   = os.environ.get("AGENT_ID", "1")  # "1" or "2"

DIRS   = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
OPP    = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
          Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT}
DIR2S  = {Direction.UP:"UP", Direction.DOWN:"DOWN",
          Direction.LEFT:"LEFT", Direction.RIGHT:"RIGHT"}

# =========================
# Zobrist hashing (fast TT)
# =========================
_rng = random.Random(20251108)
ZOBRIST = {"W":None, "H":None, "cell":[]}

def ensure_zobrist(W, H):
    if ZOBRIST["W"] == W and ZOBRIST["H"] == H and len(ZOBRIST["cell"]) == W*H:
        return
    ZOBRIST["W"], ZOBRIST["H"] = W, H
    ZOBRIST["cell"] = [_rng.getrandbits(64) for _ in range(W*H)]

def blocked_hash(blocked, W, H):
    ensure_zobrist(W, H)
    zc = ZOBRIST["cell"]; h = 0
    for (x, y) in blocked: h ^= zc[y*W + x]
    return h

# =========================
# Board helpers
# =========================
def wrap(x, y, W, H): return (x % W, y % H)

def step(pos, d, W, H):
    dx, dy = d.value
    return wrap(pos[0] + dx, pos[1] + dy, W, H)

def infer_heading(trail):
    if len(trail) < 2: return Direction.RIGHT
    (x1,y1),(x2,y2) = trail[-2], trail[-1]
    dx, dy = x2-x1, y2-y1
    if dx > 1: dx = -1
    if dx < -1: dx =  1
    if dy > 1: dy = -1
    if dy < -1: dy =  1
    for d in DIRS:
        if (dx,dy) == d.value: return d
    return Direction.RIGHT

def read_pov(player_number, state):
    board = state.get("board", [])
    H = len(board) or GLOBAL_GAME.board.height
    W = len(board[0]) if board else GLOBAL_GAME.board.width
    if player_number == 1:
        my_trail  = [tuple(p) for p in state.get("agent1_trail", [])]
        opp_trail = [tuple(p) for p in state.get("agent2_trail", [])]
        my_boosts  = int(state.get("agent1_boosts", 0))
        opp_boosts = int(state.get("agent2_boosts", 0))
    else:
        my_trail  = [tuple(p) for p in state.get("agent2_trail", [])]
        opp_trail = [tuple(p) for p in state.get("agent1_trail", [])]
        my_boosts  = int(state.get("agent2_boosts", 0))
        opp_boosts = int(state.get("agent1_boosts", 0))
    head    = my_trail[-1] if my_trail else (0,0)
    opphead = opp_trail[-1] if opp_trail else (W-1,H-1)
    my_dir  = infer_heading(my_trail)
    opp_dir = infer_heading(opp_trail)
    occupied = set(my_trail) | set(opp_trail)

    def is_free(cell):
        if board:
            x,y = cell
            return 0 <= x < W and 0 <= y < H and board[y][x] == 0
        return cell not in occupied

    return W,H,board,my_trail,opp_trail,my_boosts,opp_boosts,head,opphead,my_dir,opp_dir,occupied,is_free

def bfs_area(start, blocked, W, H, limit=None):
    if start in blocked: return 0
    q = deque([start]); seen = {start}; c = 0
    while q:
        u = q.popleft(); c += 1
        if limit is not None and c > limit: return c
        for d in DIRS:
            v = step(u,d,W,H)
            if v not in blocked and v not in seen:
                seen.add(v); q.append(v)
    return c

def torus_dist(a,b,W,H):
    ax,ay = a; bx,by = b
    dx = min(abs(ax-bx), W-abs(ax-bx))
    dy = min(abs(ay-by), H-abs(ay-by))
    return dx+dy

def voronoi_split(a, b, blocked, W, H, cap=None):
    """Rough who-can-reach-first split to shape weights."""
    qa = deque([a]); qb = deque([b])
    da={a:0}; db={b:0}
    vset=set([('a',a),('b',b)])
    ac=bc=nc=0
    while qa or qb:
        if qa:
            u=qa.popleft()
            for d in DIRS:
                v=step(u,d,W,H)
                if v in blocked or v in da: continue
                da[v]=da[u]+1; qa.append(v)
                if ('a',v) not in vset: vset.add(('a',v))
        if qb:
            u=qb.popleft()
            for d in DIRS:
                v=step(u,d,W,H)
                if v in blocked or v in db: continue
                db[v]=db[u]+1; qb.append(v)
                if ('b',v) not in vset: vset.add(('b',v))
        if cap and (ac+bc+nc)>=cap: break
    cells=set(da)|set(db)
    for c in cells:
        aa=da.get(c,10**9); bb=db.get(c,10**9)
        if aa<bb: ac+=1
        elif bb<aa: bc+=1
        else: nc+=1
    return ac,bc,nc

# =========================
# Move generation (6 actions)
# =========================
def gen_actions(head, cur_dir, boosts_left, occupied, W, H):
    """Return exactly the 6 legal first actions: 3 directions (no reverse) × {no boost, boost-if-2free}."""
    dirs = [d for d in DIRS if d != OPP[cur_dir]]
    acts = []
    for d in dirs:
        n1 = step(head, d, W, H)
        if n1 in occupied: 
            # no normal step; also no boost (requires n1 free)
            continue
        acts.append((d, False, [n1]))  # 1-step
        if boosts_left > 0:
            n2 = step(n1, d, W, H)
            if n2 not in occupied:
                acts.append((d, True, [n1, n2]))  # 2-step boost
    # Ensure we return at most 6 (3 dirs × 2 boost options)
    # If a dir has no boost option it contributes just 1.
    return acts[:6]

def gen_opp_actions(opp_head, opp_dir, opp_boosts, occupied, W, H):
    dirs = [d for d in DIRS if d != OPP[opp_dir]]
    acts = []
    for d in dirs:
        n1 = step(opp_head, d, W, H)
        if n1 in occupied: 
            continue
        acts.append((d, False, [n1]))
        if opp_boosts > 0:
            n2 = step(n1, d, W, H)
            if n2 not in occupied:
                acts.append((d, True, [n1, n2]))
    return acts

# =========================
# Simultaneous move rules
# =========================
def path_collision_us_die(my_path, opp_path):
    """Return True if OUR move would kill us in simultaneous resolution."""
    # Destination clash (opp lands on our last cell) OR cross-through on our first boost step.
    if my_path[-1] in opp_path: return True
    if len(my_path) >= 2 and my_path[0] in opp_path: return True
    return False

def apply_paths(blocked, my_path, opp_path):
    nb = blocked.copy()
    for p in my_path:  nb.add(p)
    for p in opp_path: nb.add(p)
    return nb

def has_reply(pos, blocked, W, H):
    for d in DIRS:
        if step(pos, d, W, H) not in blocked:
            return True
    return False

# =========================
# Evaluation (weights)
# =========================
def eval_position(my_pos, opp_pos, blocked, W, H, dl):
    """Higher is better for us. Penalize smaller partition traps heavily."""
    # Quick survival baseline
    my_moves  = has_reply(my_pos, blocked, W, H)
    opp_moves = has_reply(opp_pos, blocked, W, H)
    if not my_moves and not opp_moves: return -0.9   # draw ~ loss
    if not my_moves: return -1.0
    if not opp_moves: return +1.0

    # Area and partition check
    lim = 240 if dl.time_left() > 0.8 else 120
    my_area  = bfs_area(my_pos,  blocked, W, H, limit=lim)
    opp_area = bfs_area(opp_pos, blocked, W, H, limit=lim)

    # If disconnected pockets (approx): being in the smaller is very bad
    if my_area + opp_area < (W*H - len(blocked)):  # rough indicator of separation
        if my_area < opp_area: 
            return -0.8 + 0.0001*(my_area - opp_area)  # strongly avoid smaller pocket
        if my_area > opp_area:
            return  0.8 + 0.0001*(my_area - opp_area)

    # Voronoi flavor to reflect race-to-frontier
    a,b,_ = voronoi_split(my_pos, opp_pos, blocked, W, H, cap=300 if dl.time_left()>0.8 else 160)
    dist = torus_dist(my_pos, opp_pos, W, H)

    # Weighted mix
    score = 0.004*(my_area - opp_area) + 0.003*(a - b) - 0.001*dist
    return max(min(score, 0.99), -0.99)

# =========================
# 6-ply adversarial search
# =========================
DRAW_SCORE = -0.9

def order_my(acts, head, blocked, W, H, dl):
    """Rough ordering: prefer actions that increase our space."""
    scored=[]
    lim = 120 if dl.time_left() < 0.8 else 200
    for d,boost,path in acts:
        nb = blocked.copy()
        for p in path: nb.add(p)
        sc = bfs_area(path[-1], nb, W, H, limit=lim)
        scored.append((-sc, (d,boost,path)))
    scored.sort(key=lambda x:x[0])
    return [abp for _,abp in scored]

def order_opp(acts, opp_head, blocked, W, H, dl):
    # Prefer opp actions that *reduce* our space (approx by increasing theirs)
    scored=[]
    lim = 120 if dl.time_left() < 0.8 else 200
    for d,boost,path in acts:
        nb = blocked.copy()
        for p in path: nb.add(p)
        sc = bfs_area(path[-1], nb, W, H, limit=lim)
        scored.append((-sc, (d,boost,path)))
    scored.sort(key=lambda x:x[0])
    return [abp for _,abp in scored]

def solve(head, opp_head, my_dir, opp_dir, my_boosts, opp_boosts, blocked, W, H,
          depth, dl, tt):
    """Return (value, first_action or None) using AND–OR with simultaneous rules."""
    if dl.expired(): return (DRAW_SCORE, None)
    h = blocked_hash(blocked, W, H)
    key = (head, opp_head, my_dir, opp_dir, my_boosts, opp_boosts, depth, h)
    if key in tt: return tt[key]

    if depth == 0:
        val = eval_position(head, opp_head, blocked, W, H, dl)
        tt[key] = (val, None); return tt[key]

    my_acts = gen_actions(head, my_dir, my_boosts, blocked, W, H)
    if not my_acts:
        tt[key] = (-1.0, None); return tt[key]  # dead
    my_acts = order_my(my_acts, head, blocked, W, H, dl)
    # Dynamic cap
    tl = dl.time_left()
    cap_my = 6 if tl>1.2 else (4 if tl>0.7 else 3)
    my_acts = my_acts[:cap_my]

    best = -2.0; best_act=None
    for d, use_boost, my_path in my_acts:
        if dl.expired(): break
        # Opp replies enumerated on CURRENT board (no pre-block)
        opp_acts = gen_opp_actions(opp_head, opp_dir, opp_boosts, blocked, W, H)
        if not opp_acts:
            # opponent frozen: we win
            tt[key] = (1.0, (d,use_boost,my_path)); return tt[key]
        opp_acts = order_opp(opp_acts, opp_head, blocked, W, H, dl)
        cap_opp = 6 if tl>1.2 else (4 if tl>0.7 else 3)
        opp_acts = opp_acts[:cap_opp]

        worst = 2.0
        for d2, opp_boost, opp_path in opp_acts:
            # If our path collides (we die), this action is illegal/terrible: treat as loss
            if path_collision_us_die(my_path, opp_path):
                val = -1.0
            else:
                nb = apply_paths(blocked, my_path, opp_path)
                my_next  = my_path[-1]
                opp_next = opp_path[-1]
                # Immediate terminal checks
                if not has_reply(my_next, nb, W, H):
                    val = -1.0
                elif not has_reply(opp_next, nb, W, H):
                    val = 1.0
                else:
                    # Next ply: update headings conservatively as the direction taken
                    next_my_dir  = d
                    next_opp_dir = d2
                    nxt_my_boosts  = my_boosts  - (1 if use_boost else 0)
                    nxt_opp_boosts = opp_boosts - (1 if opp_boost else 0)
                    # Adaptive depth if time tight
                    sub_depth = depth-1 if dl.time_left() > 0.25 else max(1, depth-2)
                    val, _ = solve(my_next, opp_next, next_my_dir, next_opp_dir,
                                   nxt_my_boosts, nxt_opp_boosts, nb, W, H,
                                   sub_depth, dl, tt)
            if val < worst: worst = val
            if worst <= -1.0 or dl.expired(): break  # opponent has refutation

        if worst > best:
            best = worst; best_act = (d,use_boost,my_path)
            if best >= 1.0: break  # proven winning first action

    tt[key] = (best, best_act)
    return tt[key]

# =========================
# Main chooser (weights)
# =========================
def choose_weighted6(player_number, state):
    dl = Deadline(MOVE_BUDGET_SEC)

    # Read POV
    W,H,board,my_trail,opp_trail,my_boosts,opp_boosts,head,opphead,my_dir,opp_dir,occupied,is_free = read_pov(player_number, state)

    # Enumerate exactly 6 first actions (no reverse)
    first_acts = gen_actions(head, my_dir, my_boosts, occupied, W, H)
    if not first_acts:
        # No legal forward/left/right: choose any non-reverse that exists or die trying
        for d in DIRS:
            if d == OPP[my_dir]:  # explicit guard
                continue
        return {"move": DIR2S[my_dir]}  # fallback

    # For each first action, evaluate worst-case value at depth up to 6 ply
    depth = 6 if dl.time_left() > 1.6 else (5 if dl.time_left() > 1.0 else 4)
    weights = []
    tt = {}

    # Opponent could also be far: limit deep search to relevant head distances
    if torus_dist(head, opphead, W, H) > 8:
        depth = min(depth, 4)

    # Order our first actions quickly to spend time on promising ones
    first_acts = order_my(first_acts, head, occupied, W, H, dl)

    for d, boost, path in first_acts:
        if dl.expired(): break
        # Quick illegal kill check vs "do-nothing opp" path set: we still must handle simultaneous,
        # but worst-case will be handled inside solve().
        tt_local = {}
        val, _ = solve(
            head_after := path[-1],
            opphead,
            d, opp_dir,
            my_boosts - (1 if boost else 0),
            opp_boosts,
            apply_paths(occupied, path, []),
            W, H, depth-1, dl, tt_local
        )
        # Weight for THIS initial action is its worst-case line (minimax inside solve).
        # Penalize immediate self-kill if somehow not caught:
        if not has_reply(head_after, apply_paths(occupied, path, []), W, H):
            val = -1.0
        weights.append((val, (d, boost, path)))

    # Pick the highest weight found so far (ties broken by keeping straight if possible)
    if not weights:
        # If time expired very early, just pick the safest non-reverse single step
        safe = [(d,False,[step(head,d,W,H)]) for d in [dd for dd in DIRS if dd != OPP[my_dir]]
                if step(head,d,W,H) not in occupied]
        if safe:
            d,_,_ = safe[0]
            return {"move": DIR2S[d]}
        # last resort: keep direction if possible
        n1 = step(head, my_dir, W, H)
        return {"move": (DIR2S[my_dir] if n1 not in occupied else DIR2S[[d for d in DIRS if d!=OPP[my_dir]][0]])}

    weights.sort(key=lambda x: x[0], reverse=True)
    best_val, (bd, bboost, bpath) = weights[0]

    # Build final move string (never reverse — guaranteed by generator)
    move = DIR2S[bd]
    if bboost: move += ":BOOST"
    return {"move": move}

# =========================
# State sync
# =========================
def _update_local_game_from_post(data: dict):
    global GLOBAL_GAME
    LAST_POSTED_STATE.update(data)
    H = int(data.get("board_height", getattr(GLOBAL_GAME.board, "height", 18)))
    W = int(data.get("board_width",  getattr(GLOBAL_GAME.board, "width", 20)))
    if (H,W) != (GLOBAL_GAME.board.height, GLOBAL_GAME.board.width):
        GLOBAL_GAME = Game()
        GLOBAL_GAME.board.height = H
        GLOBAL_GAME.board.width  = W
    if "board" in data and data["board"]:
        GLOBAL_GAME.board.grid = data["board"]
    if "agent1_trail" in data:
        GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"])
    if "agent2_trail" in data:
        GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"])
    if "agent1_boosts" in data:
        GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
    if "agent2_boosts" in data:
        GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
    if "agent1_alive" in data:
        GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
    if "agent2_alive" in data:
        GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
    if "turn_count" in data:
        GLOBAL_GAME.turns = int(data["turn_count"])

# =========================
# HTTP routes
# =========================
@app.get("/")
def meta():
    return jsonify({"status":"ok","agent_name":AGENT_NAME,"agent_id":AGENT_ID}), 200

@app.route("/send-state", methods=["GET","POST"])
def receive_state():
    data = request.get_json(silent=True) or {}
    with game_lock:
        _update_local_game_from_post(data)
    return jsonify({"status":"received"}), 200

@app.route("/send-move", methods=["GET","POST"])
def send_move():
    if request.method == "GET":
        player_number = int(request.args.get("player_number", 1))
        deltas = {}
        if "turn_count" in request.args:
            deltas["turn_count"] = int(request.args.get("turn_count"))
    else:
        payload = request.get_json(silent=True) or {}
        player_number = int(payload.get("player_number", 1))
        deltas = {k:v for k,v in payload.items() if k not in ("player_number","attempt_number")}
    with game_lock:
        state = dict(LAST_POSTED_STATE)
        state.update(deltas)
        res = choose_weighted6(player_number, state)
    return jsonify(res), 200

@app.post("/end-game")
def end_game():
    data = request.get_json(silent=True) or {}
    with game_lock:
        _update_local_game_from_post(data)
    return jsonify({"status":"acknowledged"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT","5009"))
    app.run(host="0.0.0.0", port=port, debug=True)
