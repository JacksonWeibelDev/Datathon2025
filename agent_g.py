import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult


#new imports
import time
import math
from copy import deepcopy



# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
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
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


# --- Direction helpers ---
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
    """Return list of safe (non-crashing, non-backward) directions for the given agent."""
    safe_moves = []
    head = agent.trail[-1]
    prev = agent.trail[-2] if len(agent.trail) > 1 else None

    for name, direction in DIRECTIONS.items():
        dx, dy = direction.value
        new_pos = ((head[0] + dx) % board.width, (head[1] + dy) % board.height)

        # Don't go backward
        if prev and new_pos == prev:
            continue

        # Check for collisions (self or opponent)
        if new_pos in agent.trail or new_pos in other_agent.trail:
            continue  # unsafe move

        safe_moves.append(name)

    # If no safe moves exist, return all possible (we’re trapped) -> return DIR_ORDER
    # (previous version attempted to filter by prev incorrectly)
    return safe_moves if safe_moves else DIR_ORDER[:]




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
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
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

def minimax(game, depth, alpha, beta, maximizing, start_time, time_limit):
    """Minimax with alpha-beta pruning, safety checks, and time cutoff."""
    if time.time() - start_time > time_limit:
        raise TimeoutError

    if depth == 0 or not (game.agent1.alive and game.agent2.alive):
        return evaluate_state(game), None

    best_move = None

    if maximizing:
        max_eval = -math.inf
        head_agent = game.agent1
        opp_agent = game.agent2

        possible_moves = [m for m in DIR_ORDER if is_safe_move(head_agent, opp_agent, game.board, m)]
        if not possible_moves:
            possible_moves = DIR_ORDER[:]  # forced

        for move_name in possible_moves:
            new_game = deepcopy(game)
            new_game.agent1.move(DIRECTIONS[move_name], other_agent=new_game.agent2)
            eval_score, _ = minimax(new_game, depth-1, alpha, beta, False, start_time, time_limit)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move_name
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move

    else:
        min_eval = math.inf
        head_agent = game.agent2
        opp_agent = game.agent1

        possible_moves = [m for m in DIR_ORDER if is_safe_move(head_agent, opp_agent, game.board, m)]
        if not possible_moves:
            possible_moves = DIR_ORDER[:]  # forced

        for move_name in possible_moves:
            new_game = deepcopy(game)
            new_game.agent2.move(DIRECTIONS[move_name], other_agent=new_game.agent1)
            eval_score, _ = minimax(new_game, depth-1, alpha, beta, True, start_time, time_limit)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move_name
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move



@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)
        # original agents from GLOBAL_GAME (do not modify these)
        orig_my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        orig_opponent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        boosts_remaining = orig_my_agent.boosts_remaining
        # make a deep copy to simulate safely
        local_game = deepcopy(GLOBAL_GAME)

    # If the requester is player 2, swap agent1/agent2 in the local_game so minimax can always maximize "agent1"
    if player_number == 2:
        # swap agents in local_game
        tmp_a1 = local_game.agent1
        local_game.agent1 = local_game.agent2
        local_game.agent2 = tmp_a1

    # Now local_game.agent1 corresponds to the requesting player
    local_my_agent = local_game.agent1
    local_opponent = local_game.agent2

    # --- Minimax decision ---
    depth_limit = 3
    time_limit = 0.15  # 150ms max think time

    try:
        start = time.time()
        _, best_move = minimax(local_game, depth_limit, -math.inf, math.inf, True, start, time_limit)

        # Build safe move set for the local (requesting) agent in the simulated game
        safe_moves = get_safe_moves(local_my_agent, local_opponent, local_game.board)

        # If minimax returned nothing, or returned a move that is not safe, pick a safe move instead.
        if not best_move or best_move not in safe_moves:
            # prefer the first safe move; if none, use DIR_ORDER fallback
            move = safe_moves[0] if safe_moves else DIR_ORDER[0]
        else:
            move = best_move

    except TimeoutError:
        # on timeout, pick a safe move from the local simulated state
        safe_moves = get_safe_moves(local_my_agent, local_opponent, local_game.board)
        move = safe_moves[0] if safe_moves else DIR_ORDER[0]

    # move is a string in {"UP","DOWN","LEFT","RIGHT"}; return as required
    return jsonify({"move": move}), 200



@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
