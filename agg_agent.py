import os
from collections import deque
from threading import Lock
from flask import Flask, request, jsonify

from case_closed_game import Game, Direction

# =========================
# Flask + Global State
# =========================
app = Flask(__name__)
game_lock = Lock()

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

AGENT_NAME = os.environ.get("AGENT_NAME", "Bully3BoostFinisher")
AGENT_ID = os.environ.get("AGENT_ID", "1")  # string "1" or "2"

DIRS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
OPP = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

# =========================
# Utilities
# =========================
def wrap(x, y, W, H):
    return (x % W, y % H)

def step(pos, d, W, H):
    dx, dy = d.value
    return wrap(pos[0] + dx, pos[1] + dy, W, H)

def infer_heading(trail):
    if len(trail) < 2:
        return Direction.RIGHT
    (x1, y1), (x2, y2) = trail[-2], trail[-1]
    dx, dy = x2 - x1, y2 - y1
    if dx > 1: dx = -1
    if dx < -1: dx = 1
    if dy > 1: dy = -1
    if dy < -1: dy = 1
    for d in DIRS:
        if (dx, dy) == d.value:
            return d
    return Direction.RIGHT

def read_pov(player_number, state):
    board = state.get("board", [])
    H = len(board) or GLOBAL_GAME.board.height
    W = len(board[0]) if board else GLOBAL_GAME.board.width

    if player_number == 1:
        my_trail  = [tuple(p) for p in state.get("agent1_trail", [])]
        opp_trail = [tuple(p) for p in state.get("agent2_trail", [])]
        my_boosts = int(state.get("agent1_boosts", 0))
        opp_boosts = int(state.get("agent2_boosts", 0))
    else:
        my_trail  = [tuple(p) for p in state.get("agent2_trail", [])]
        opp_trail = [tuple(p) for p in state.get("agent1_trail", [])]
        my_boosts = int(state.get("agent2_boosts", 0))
        opp_boosts = int(state.get("agent1_boosts", 0))

    head = my_trail[-1] if my_trail else (0, 0)
    cur_dir = infer_heading(my_trail)

    occupied = set(my_trail) | set(opp_trail)

    def is_free(cell):
        if board:
            x, y = cell
            return 0 <= x < W and 0 <= y < H and board[y][x] == 0
        return cell not in occupied

    return W, H, board, my_trail, opp_trail, my_boosts, opp_boosts, head, cur_dir, occupied, is_free

def corridor_len(head, d, is_free, W, H, limit=3):
    pos = head
    cnt = 0
    for _ in range(limit):
        pos = step(pos, d, W, H)
        if not is_free(pos): break
        cnt += 1
    return cnt

def bfs_area(start, blocked, W, H, limit=None):
    """Flood-fill reachable area on torus; blocked is a set of occupied cells."""
    if start in blocked:
        return 0
    q = deque([start])
    seen = {start}
    c = 0
    while q:
        u = q.popleft()
        c += 1
        if limit is not None and c > limit:
            return c
        for d in DIRS:
            v = step(u, d, W, H)
            if v not in seen and v not in blocked:
                seen.add(v)
                q.append(v)
    return c

def voronoi_score(my_head, opp_head, blocked, W, H, cap=None):
    """Approximate split of free cells by multi-source BFS distances."""
    from collections import deque as _dq
    q = _dq()
    dist_me, dist_opp = {}, {}

    if my_head not in blocked:
        q.append(('me', my_head)); dist_me[my_head] = 0
    if opp_head not in blocked:
        q.append(('opp', opp_head)); dist_opp[opp_head] = 0

    visited = set()
    while q:
        tag, u = q.popleft()
        if (tag, u) in visited: continue
        visited.add((tag, u))
        for d in DIRS:
            v = step(u, d, W, H)
            if v in blocked: continue
            if tag == 'me':
                nv = dist_me[u] + 1
                if v not in dist_me or nv < dist_me[v]:
                    dist_me[v] = nv
                    q.append(('me', v))
            else:
                nv = dist_opp[u] + 1
                if v not in dist_opp or nv < dist_opp[v]:
                    dist_opp[v] = nv
                    q.append(('opp', v))

    me_cells = opp_cells = neutral = 0
    all_pos = set(dist_me) | set(dist_opp)
    for cell in all_pos:
        dm = dist_me.get(cell, 10**9)
        do = dist_opp.get(cell, 10**9)
        if dm < do: me_cells += 1
        elif do < dm: opp_cells += 1
        else: neutral += 1
        if cap and (me_cells + opp_cells + neutral) >= cap:
            break
    return me_cells, opp_cells, neutral

def torus_dist(a, b, W, H):
    ax, ay = a; bx, by = b
    dx = min(abs(ax - bx), W - abs(ax - bx))
    dy = min(abs(ay - by), H - abs(ay - by))
    return dx + dy  # Manhattan on torus

# ---------- Paths & Forced-Boost Helpers ----------
def legal_steps_from(pos, blocked, W, H):
    out = []
    for d in DIRS:
        m1 = step(pos, d, W, H)
        if m1 not in blocked:
            out.append((d, [m1]))  # 1-step path
    return out

def legal_steps_with_boost_from(pos, boosts_left, blocked, W, H):
    out = legal_steps_from(pos, blocked, W, H)
    if boosts_left > 0:
        ext = []
        for d, path in out:
            m1 = path[0]
            m2 = step(m1, d, W, H)
            if m2 not in blocked:
                ext.append((d, [m1, m2]))  # 2-step (BOOST)
        out = out + ext
    return out

def next_blocked_after(path_me, path_opp, blocked):
    nb = set(blocked)
    for p in path_me:  nb.add(p)
    for p in path_opp: nb.add(p)
    return nb

def path_collides_simultaneous(path_me, path_opp):
    """Veto if opponent ends on our last cell (dest clash) or crosses our first boost cell."""
    last_me = path_me[-1]
    if last_me in path_opp:
        return True
    if len(path_me) >= 2 and path_me[0] in path_opp:
        return True
    return False

def any_legal_from(pos, blocked, W, H):
    for d in DIRS:
        if step(pos, d, W, H) not in blocked:
            return True
    return False

def can_force_win_boost_chain(my_pos, opp_pos, my_boosts_left, opp_boosts_left, blocked, W, H, turns_left, prev_dir=None):
    """
    Our side to move. We must BOOST every turn (while turns_left>0 and boosts remain).
    Return the first direction that guarantees a kill in <= turns_left boosts,
    i.e., for ALL opponent replies (with/without boost), the line still forces loss for them.
    """
    if turns_left == 0 or my_boosts_left <= 0:
        return False

    # Only BOOST actions; avoid immediate reverse of previous direction for smoother chains.
    boost_actions = []
    for d in DIRS:
        if prev_dir is not None and d == OPP[prev_dir]:
            continue
        m1 = step(my_pos, d, W, H)
        if m1 in blocked:
            continue
        m2 = step(m1, d, W, H)
        if m2 in blocked:
            continue
        boost_actions.append((d, [m1, m2]))

    if not boost_actions:
        return False

    for d, my_path in boost_actions:
        # Opponent replies are generated on CURRENT occupied (no pre-block with our path)
        opp_opts = legal_steps_with_boost_from(opp_pos, opp_boosts_left, blocked, W, H)
        if not opp_opts:
            # Opponent frozen immediately → win
            return d

        all_branches_win = True
        for _, opp_path in opp_opts:
            # Simultaneous clash awareness
            if path_collides_simultaneous(my_path, opp_path):
                all_branches_win = False
                break

            nb = next_blocked_after(my_path, opp_path, blocked)
            my_next  = my_path[-1]
            opp_next = opp_path[-1]

            # Opponent dead next turn → this branch is a win
            if not any_legal_from(opp_next, nb, W, H):
                continue

            # Continue forcing chain
            if not can_force_win_boost_chain(
                my_next, opp_next,
                my_boosts_left - 1,
                max(0, opp_boosts_left - (1 if len(opp_path) == 2 else 0)),
                nb, W, H,
                turns_left - 1,
                prev_dir=d
            ):
                all_branches_win = False
                break

        if all_branches_win:
            return d

    return False

# =========================
# Bully Move Chooser
# =========================
def choose_bully_move(player_number, state):
    """
    Bully/partitioner with:
      - No reverse.
      - Safety-first scoring (our area vs opp Voronoi) + corridor bias.
      - Adaptive distance weight: small when close, large when far.
      - Forced 3-BOOST finisher when close and holding ≥3 boosts.
      - Conservative BOOST saver otherwise (simultaneous-collision aware).
    """
    W, H, board, my_trail, opp_trail, my_boosts, opp_boosts, head, cur_dir, occupied, is_free = read_pov(player_number, state)
    opp_head = opp_trail[-1] if opp_trail else head

    # ===== Try forced 3-BOOST win chain first =====
    PROX_FORCE = 4   # try when already nearby
    MAX_CHAIN  = 3   # use all 3 boosts
    if my_boosts >= 3:
        if torus_dist(head, opp_head, W, H) <= PROX_FORCE:
            forced_dir = can_force_win_boost_chain(
                head, opp_head,
                my_boosts_left=my_boosts,
                opp_boosts_left=opp_boosts,
                blocked=occupied,
                W=W, H=H,
                turns_left=MAX_CHAIN
            )
            if forced_dir:
                return {"move": {Direction.UP:"UP", Direction.DOWN:"DOWN",
                                 Direction.LEFT:"LEFT", Direction.RIGHT:"RIGHT"}[forced_dir] + ":BOOST"}

    # ===== Standard bully scoring (no reverse) =====
    cands = [d for d in DIRS if d != OPP[cur_dir]]
    cands = [d for d in cands if is_free(step(head, d, W, H))]

    if not cands:
        # If nothing safe, still avoid reverse if possible; pick least-bad by 1-step corridor
        fallback = [d for d in DIRS if d != OPP[cur_dir]]
        fallback.sort(key=lambda d: corridor_len(head, d, is_free, W, H, limit=1), reverse=True)
        chosen = fallback[0]
        if chosen == OPP[cur_dir]:
            nonrev = [d for d in fallback if d != OPP[cur_dir]]
            if nonrev:
                chosen = nonrev[0]
    else:
        # Safety weights
        K_OPP    = 1.6
        K_NEUT   = 0.2
        K_CORR   = 0.06
        STRAIGHT = 0.5

        # Distance weight ramp
        CLOSE_R = 5
        FAR_R   = 8
        W_MIN   = -1.0
        W_MAX   = 2.0
        def dist_weight(d_now):
            if d_now <= CLOSE_R: return W_MIN
            if d_now >= FAR_R:   return W_MAX
            t = (d_now - CLOSE_R) / float(FAR_R - CLOSE_R)
            return W_MIN + t * (W_MAX - W_MIN)

        best, best_val = None, -1e9
        for d in cands:
            n_my = step(head, d, W, H)
            blocked_next = set(occupied); blocked_next.add(n_my)

            me_cells, opp_cells, neutral = voronoi_score(n_my, opp_head, blocked_next, W, H, cap=220)
            my_area_est = bfs_area(n_my, blocked_next, W, H, limit=220)

            safety_score = (my_area_est - K_OPP * opp_cells) + K_NEUT * neutral
            safety_score += K_CORR * corridor_len(head, d, is_free, W, H, limit=3)
            if d == cur_dir:
                safety_score += STRAIGHT

            d_after = torus_dist(n_my, opp_head, W, H)
            score = safety_score + dist_weight(d_after) * ( - float(d_after) )

            if score > best_val:
                best_val, best = score, d

        chosen = best
        if chosen == OPP[cur_dir]:
            nonrev = [d for d in cands if d != OPP[cur_dir]]
            if nonrev:
                chosen = nonrev[0]

    move = {Direction.UP: "UP", Direction.DOWN: "DOWN",
            Direction.LEFT: "LEFT", Direction.RIGHT: "RIGHT"}[chosen]

    # ===== Conservative BOOST saver (simul-collision aware) =====
    PROX = 3          # only consider boosting if already close
    AREA_MARGIN = 10  # require small edge after replies

    def safe_cell(pos):
        if board:
            x, y = pos
            return 0 <= x < W and 0 <= y < H and board[y][x] == 0
        return pos not in occupied

    def area_eval(my_pos, opp_pos, blocked):
        my_a  = bfs_area(my_pos, blocked, W, H, limit=300)
        opp_a = bfs_area(opp_pos, blocked, W, H, limit=300)
        return my_a - opp_a

    def boost_is_safe_simul(n1, n2, opp_head, opp_boosts):
        # Opponent replies enumerated on current occupied (no pre-block with our path)
        opp_opts = legal_steps_with_boost_from(opp_head, opp_boosts, occupied, W, H)
        if not opp_opts:
            return True

        our_path = [n1, n2]
        for _, opp_path in opp_opts:
            # Simultaneous collision checks
            if path_collides_simultaneous(our_path, opp_path):
                return False

            # Build next blocked after both paths
            blocked_next = next_blocked_after(our_path, opp_path, occupied)

            # We must have at least one reply
            if not any_legal_from(n2, blocked_next, W, H):
                return False

            # Keep a small area edge
            if area_eval(n2, opp_path[-1], blocked_next) < AREA_MARGIN:
                return False

        return True

    if my_boosts > 0:
        n1 = step(head, chosen, W, H)
        n2 = step(n1, chosen, W, H)
        close_now = torus_dist(head, opp_head, W, H) <= PROX

        if close_now and safe_cell(n1) and safe_cell(n2):
            # be conservative about landing exactly on current opp head
            if n2 != opp_head and boost_is_safe_simul(n1, n2, opp_head, opp_boosts):
                move = f"{move}:BOOST"

    return {"move": move}

# =========================
# State Sync Helpers
# =========================
def _update_local_game_from_post(data: dict):
    global GLOBAL_GAME
    LAST_POSTED_STATE.update(data)

    H = int(data.get("board_height", getattr(GLOBAL_GAME.board, "height", 18)))
    W = int(data.get("board_width", getattr(GLOBAL_GAME.board, "width", 20)))
    if (H, W) != (GLOBAL_GAME.board.height, GLOBAL_GAME.board.width):
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
# HTTP Routes
# =========================
@app.get("/")
def meta():
    return jsonify({"status": "ok", "agent_name": AGENT_NAME, "agent_id": AGENT_ID}), 200

@app.route("/send-state", methods=["GET", "POST"])
def receive_state():
    data = request.get_json(silent=True) or {}
    with game_lock:
        _update_local_game_from_post(data)
    return jsonify({"status": "received"}), 200

@app.route("/send-move", methods=["GET", "POST"])
def send_move():
    if request.method == "GET":
        player_number = int(request.args.get("player_number", 1))
        deltas = {}
        if "turn_count" in request.args:
            deltas["turn_count"] = int(request.args.get("turn_count"))
    else:
        payload = request.get_json(silent=True) or {}
        player_number = int(payload.get("player_number", 1))
        deltas = {k: v for k, v in payload.items()
                  if k not in ("player_number", "attempt_number")}

    with game_lock:
        state = dict(LAST_POSTED_STATE)
        state.update(deltas)
        res = choose_bully_move(player_number, state)

    return jsonify(res), 200

@app.post("/end-game")
def end_game():
    data = request.get_json(silent=True) or {}
    with game_lock:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
