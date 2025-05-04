import numpy as np
import random
import math
import copy
from collections import defaultdict

# Constants for the game
ROW_COUNT = 6
COLUMN_COUNT = 7
WINDOW_LENGTH = 4  # Number of discs in a row needed to win
EMPTY = 0
PLAYER_PIECE = 1  # Agent's piece
OPPONENT_PIECE = 2  # Opponent's piece

# Define Agent Types
AGENT_A = 'AgentA'  # Minimax Agent
AGENT_B = 'AgentB'  # Expectimax Agent
AGENT_C = 'AgentC'  # Random Agent

# Probability for AgentB (set as needed)
PROBABILITY_P = 0.5  # Change as required

# Depth for search algorithms
SEARCH_DEPTH = 4  # Adjust based on computational resources

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == EMPTY

def get_valid_locations(board):
    return [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == EMPTY:
            return r

def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    # Horizontal check
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if all([board[r][c+i] == piece for i in range(WINDOW_LENGTH)]):
                return True
    # Vertical check
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all([board[r+i][c] == piece for i in range(WINDOW_LENGTH)]):
                return True
    # Positive diagonal check
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if all([board[r+i][c+i] == piece for i in range(WINDOW_LENGTH)]):
                return True
    # Negative diagonal check
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if all([board[r-i][c+i] == piece for i in range(WINDOW_LENGTH)]):
                return True
    return False

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, OPPONENT_PIECE) or len(get_valid_locations(board)) == 0

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE if piece == OPPONENT_PIECE else OPPONENT_PIECE

    if window.count(piece) == 4:
        score += 10000
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 100
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 10
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 80  # Block opponent's winning move
    return score

def score_position(board, piece):
    score = 0
    # Center column preference
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 6

    # Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score Positive Diagonals
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Score Negative Diagonals
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def minimax(board, depth, alpha, beta, maximizingPlayer, prune_stats):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, PLAYER_PIECE):
                return (None, float('inf'))
            elif winning_move(board, OPPONENT_PIECE):
                return (None, float('-inf'))
            else:
                return (None, 0)
        else:
            return (None, score_position(board, PLAYER_PIECE))
    if maximizingPlayer:
        value = float('-inf')
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = copy.deepcopy(board)
            drop_piece(temp_board, row, col, PLAYER_PIECE)
            new_score = minimax(temp_board, depth - 1, alpha, beta, False, prune_stats)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                prune_stats['beta'] += 1  # Beta pruning
                break
        return best_col, value
    else:
        value = float('inf')
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = copy.deepcopy(board)
            drop_piece(temp_board, row, col, OPPONENT_PIECE)
            new_score = minimax(temp_board, depth - 1, alpha, beta, True, prune_stats)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                prune_stats['alpha'] += 1  # Alpha pruning
                break
        return best_col, value

def expectimax(board, depth, maximizingPlayer, p):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, PLAYER_PIECE):
                return (None, float('inf'))
            elif winning_move(board, OPPONENT_PIECE):
                return (None, float('-inf'))
            else:
                return (None, 0)
        else:
            return (None, score_position(board, PLAYER_PIECE))
    if maximizingPlayer:
        value = float('-inf')
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = copy.deepcopy(board)
            drop_piece(temp_board, row, col, PLAYER_PIECE)
            new_score = expectimax(temp_board, depth - 1, False, p)[1]
            if new_score > value:
                value = new_score
                best_col = col
        return best_col, value
    else:
        value = 0
        for col in valid_locations:
            probability = p / len(valid_locations) if is_best_move(board, col, OPPONENT_PIECE) else (1 - p) / (len(valid_locations) - 1)
            row = get_next_open_row(board, col)
            temp_board = copy.deepcopy(board)
            drop_piece(temp_board, row, col, OPPONENT_PIECE)
            new_score = expectimax(temp_board, depth - 1, True, p)[1]
            value += probability * new_score
        return None, value

def is_best_move(board, col, piece):
    # Simple heuristic to determine if a move is the best move
    row = get_next_open_row(board, col)
    temp_board = copy.deepcopy(board)
    drop_piece(temp_board, row, col, piece)
    return winning_move(temp_board, piece)

def random_agent_move(board):
    valid_locations = get_valid_locations(board)
    return random.choice(valid_locations)

def play_game(agent1, agent2, p=0.5, depth=SEARCH_DEPTH):
    board = create_board()
    game_over = False
    turn = random.randint(PLAYER_PIECE, OPPONENT_PIECE)  # Randomly select who starts
    move_count = 0
    prune_stats = {'alpha': 0, 'beta': 0}
    while not game_over:
        # Agent1's Turn
        if turn == PLAYER_PIECE:
            if agent1 == AGENT_A:
                col, minimax_score = minimax(board, depth, float('-inf'), float('inf'), True, prune_stats)
            elif agent1 == AGENT_B:
                col, expectimax_score = expectimax(board, depth, True, p)
            elif agent1 == AGENT_C:
                col = random_agent_move(board)
            else:
                col = random_agent_move(board)  # Default to random if unknown
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)
                move_count += 1
                if winning_move(board, PLAYER_PIECE):
                    return {'winner': agent1, 'moves': move_count, 'prune_stats': prune_stats}
                turn = OPPONENT_PIECE  # Switch turns
        # Agent2's Turn
        else:
            if agent2 == AGENT_A:
                col, minimax_score = minimax(board, depth, float('-inf'), float('inf'), True, prune_stats)
            elif agent2 == AGENT_B:
                col, expectimax_score = expectimax(board, depth, True, p)
            elif agent2 == AGENT_C:
                col = random_agent_move(board)
            else:
                col = random_agent_move(board)  # Default to random if unknown
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, OPPONENT_PIECE)
                move_count += 1
                if winning_move(board, OPPONENT_PIECE):
                    return {'winner': agent2, 'moves': move_count, 'prune_stats': prune_stats}
                turn = PLAYER_PIECE  # Switch turns
        if len(get_valid_locations(board)) == 0:
            return {'winner': 'Draw', 'moves': move_count, 'prune_stats': prune_stats}
    return {'winner': 'Draw', 'moves': move_count, 'prune_stats': prune_stats}

def run_games(agent1, agent2, num_games=10, p=0.5, depth=SEARCH_DEPTH):
    results = []
    for _ in range(num_games):
        result = play_game(agent1, agent2, p, depth)
        results.append(result)
    return results

def analyze_results(results, agent1, agent2):
    wins = {agent1: 0, agent2: 0, 'Draw': 0}
    move_counts = []
    alpha_prunes = []
    beta_prunes = []
    for result in results:
        wins[result['winner']] += 1
        move_counts.append(result['moves'])
        alpha_prunes.append(result['prune_stats'].get('alpha', 0))
        beta_prunes.append(result['prune_stats'].get('beta', 0))
    total_games = len(results)
    win_loss_ratio = wins[agent1] / max(wins[agent2], 1)  # Avoid division by zero
    move_stats = {
        'min': min(move_counts),
        'max': max(move_counts),
        'average': sum(move_counts) / total_games,
        'std_dev': np.std(move_counts)
    }
    prune_stats = {
        'alpha': {
            'min': min(alpha_prunes),
            'max': max(alpha_prunes),
            'average': sum(alpha_prunes) / total_games,
            'std_dev': np.std(alpha_prunes)
        },
        'beta': {
            'min': min(beta_prunes),
            'max': max(beta_prunes),
            'average': sum(beta_prunes) / total_games,
            'std_dev': np.std(beta_prunes)
        }
    }
    return {
        'wins': wins,
        'win_loss_ratio': win_loss_ratio,
        'move_stats': move_stats,
        'prune_stats': prune_stats
    }

# Example usage:
if __name__ == "__main__":
    # Set parameters as required
    AGENT1 = AGENT_A
    AGENT2 = AGENT_C
    P = 0.5  # Probability for AgentB, if applicable
    NUM_GAMES = 10
    DEPTH = 4

    # Run games
    game_results = run_games(AGENT1, AGENT2, NUM_GAMES, P, DEPTH)

    # Analyze results
    analysis = analyze_results(game_results, AGENT1, AGENT2)

    # Print analysis
    print(f"Results between {AGENT1} and {AGENT2}:")
    print(f"Wins: {analysis['wins']}")
    print(f"Win/Loss Ratio: {analysis['win_loss_ratio']}")
    print(f"Move Statistics: {analysis['move_stats']}")
    print(f"Pruning Statistics: {analysis['prune_stats']}")
