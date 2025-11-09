import os
import uuid
import time
import math
import random
from copy import deepcopy
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult


# ------------------- Flask Setup -------------------
app = Flask(__name__)
GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


# ------------------- Game Update -------------------
def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using JSON posted by the judge."""
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

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
    """Judge calls this to push the current game state."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


# ------------------- Direction Helpers -------------------
DIRECTIONS = {
    "UP": Direction.UP,
    "DOWN": Direction.DOWN,
    "LEFT": Direction.LEFT,
    "RIGHT": Direction.RIGHT
}
DIR_ORDER = ["UP", "DOWN", "LEFT", "RIGHT"]


def is_safe_move(agent, other_agent, board, move_name):
    """Return True if the move does not cause an immediate crash or reversal."""
    dx, dy = DIRECTIONS[move_name].value
    head = agent.trail[-1]
    prev = agent.trail[-2] if len(agent.trail) > 1 else None
    new_pos = ((head[0] + dx) % board.width, (head[1] + dy) % board.height)

    # Avoid moving backward
    if prev and new_pos == prev:
        return False

    # Check for collisions
    if new_pos in agent.trail or new_pos in other_agent.trail:
        return False

    return True


def get_safe_moves(agent, other_agent, board):
    """Return list of safe (non-crashing, non-backward) directions."""
    safe_moves = []
    head = agent.trail[-1]
    prev = agent.trail[-2] if len(agent.trail) > 1 else None

    for name, direction in DIRECTIONS.items():
        dx, dy = direction.value
        new_pos = ((head[0] + dx) % board.width, (head[1] + dy) % board.height)

        if prev and new_pos == prev:
            continue
        if new_pos in agent.trail or new_pos in other_agent.trail:
            continue
        safe_moves.append(name)

    return safe_moves if safe_moves else DIR_ORDER[:]


# ------------------- Evaluation -------------------
def evaluate_state(game):
    """Simple evaluation: difference in available space (flood-fill)."""
    def flood_fill(start, blocked):
        stack = [start]
        visited = set()
        while stack:
            x, y = stack.pop()
            if (x, y) in visited or (x, y) in blocked:
                continue
            visited.add((x, y))
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = (x + dx) % game.board.width, (y + dy) % game.board.height
                if (nx, ny) not in blocked:
                    stack.append((nx, ny))
        return len(visited)

    blocked = set(game.agent1.trail) | set(game.agent2.trail)
    if not game.agent1.alive:
        return -9999
    if not game.agent2.alive:
        return 9999

    a1_space = flood_fill(game.agent1.trail[-1], blocked)
    a2_space = flood_fill(game.agent2.trail[-1], blocked)
    return a1_space - a2_space


# ------------------- Minimax -------------------
def minimax(game, depth, alpha, beta, maximizing, start_time, time_limit):
    """Minimax with alpha-beta pruning and time cutoff."""
    if time.time() - start_time > time_limit:
        raise TimeoutError

    if depth == 0 or not (game.agent1.alive and game.agent2.alive):
        return evaluate_state(game), None

    best_move = None

    if maximizing:
        max_eval = -math.inf
        agent = game.agent1
        opponent = game.agent2
        possible_moves = [m for m in DIR_ORDER if is_safe_move(agent, opponent, game.board, m)] or DIR_ORDER[:]

        for move_name in possible_moves:
            new_game = deepcopy(game)
            new_game.agent1.move(DIRECTIONS[move_name], other_agent=new_game.agent2)
            eval_score, _ = minimax(new_game, depth - 1, alpha, beta, False, start_time, time_limit)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move_name
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move

    else:
        min_eval = math.inf
        agent = game.agent2
        opponent = game.agent1
        possible_moves = [m for m in DIR_ORDER if is_safe_move(agent, opponent, game.board, m)] or DIR_ORDER[:]

        for move_name in possible_moves:
            new_game = deepcopy(game)
            new_game.agent2.move(DIRECTIONS[move_name], other_agent=new_game.agent1)
            eval_score, _ = minimax(new_game, depth - 1, alpha, beta, True, start_time, time_limit)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move_name
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move


# ------------------- Move Request -------------------
@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        local_game = deepcopy(GLOBAL_GAME)

    if player_number == 2:
        local_game.agent1, local_game.agent2 = local_game.agent2, local_game.agent1

    my_agent = local_game.agent1
    opponent = local_game.agent2

    # --- Decision Phase ---
    depth_limit = 3
    time_limit = 0.15

    try:
        start = time.time()
        _, best_move = minimax(local_game, depth_limit, -math.inf, math.inf, True, start, time_limit)
        move = best_move or "UP"
    except TimeoutError:
        move = "UP"

    # --- Debug Split Detection ---
    move_offsets = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

    def flood_fill_area(board, start, blocked):
        if start in blocked:
            return 0
        from collections import deque
        q = deque([start])
        visited = set()
        while q:
            x, y = q.popleft()
            if (x, y) in visited or (x, y) in blocked:
                continue
            visited.add((x, y))
            for dx, dy in move_offsets.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < board.width and 0 <= ny < board.height:
                    if (nx, ny) not in visited and (nx, ny) not in blocked:
                        q.append((nx, ny))
        return len(visited)

    def is_blocked(board, pos, blocked):
        x, y = pos
        if not (0 <= x < board.width and 0 <= y < board.height):
            return True
        return pos in blocked

    head = my_agent.trail[-1]
    prev = my_agent.trail[-2] if len(my_agent.trail) > 1 else None
    came_from = None
    if prev:
        dx, dy = head[0] - prev[0], head[1] - prev[1]
        for name, (mx, my) in move_offsets.items():
            if (mx, my) == (dx, dy):
                came_from = name
                break

    if came_from:
        opposite_dir = opposites[came_from]
        ox, oy = move_offsets[opposite_dir]
        blocked_cells = set(local_game.agent1.trail + local_game.agent2.trail)
        opp_pos = (head[0] + ox, head[1] + oy)
        if is_blocked(local_game.board, opp_pos, blocked_cells):
            if came_from in ("LEFT", "RIGHT"):
                check_dirs = ["UP", "DOWN"]
            else:
                check_dirs = ["LEFT", "RIGHT"]

            print(f"[DEBUG] Backward ({opposite_dir}) blocked at {opp_pos}")
            for d in check_dirs:
                dx, dy = move_offsets[d]
                test_pos = (head[0] + dx, head[1] + dy)
                area = flood_fill_area(local_game.board, test_pos, blocked_cells)
                print(f"[DEBUG] Flood-fill area if moving {d}: {area}")
            # --- Choose direction with larger flood-fill area ---
            areas = {}
            for d in check_dirs:
                dx, dy = move_offsets[d]
                test_pos = (head[0] + dx, head[1] + dy)
                areas[d] = flood_fill_area(local_game.board, test_pos, blocked_cells)

            if areas:
                best_dir = max(areas, key=areas.get)
                print(f"[DEBUG] Choosing {best_dir} (larger open area: {areas[best_dir]})")
                move = best_dir

    return jsonify({"move": move}), 200


# ------------------- End Game -------------------
@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished."""
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


# ------------------- Run Server -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
