import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class TicTacToeAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=1.0):
        self.q_values = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, board):
        return tuple(board.flatten())

    def get_available_actions(self, board):
        return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

    def choose_action(self, board):
        if np.random.rand() < self.epsilon or self.get_state_key(board) not in self.q_values:
            return random.choice(self.get_available_actions(board))
        else:
            return max(self.get_available_actions(board), key=lambda a: self.q_values.get((self.get_state_key(board), a), 0))

    def update_q_values(self, state, action, reward, next_state):
        current_q = self.q_values.get((state, action), 0)
        max_next_q = max([self.q_values.get((tuple(next_state.flatten()), a), 0) for a in self.get_available_actions(next_state)])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_values[(state, action)] = new_q

    def save_q_values(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.q_values, file)

    def load_q_values(self, filename):
        with open(filename, 'rb') as file:
            self.q_values = pickle.load(file)

def check_winner(board):
    for i in range(3):
        if board[i, 0] == board[i, 1] == board[i, 2] != 0 or board[0, i] == board[1, i] == board[2, i] != 0:
            return True
    if board[0, 0] == board[1, 1] == board[2, 2] != 0 or board[0, 2] == board[1, 1] == board[2, 0] != 0:
        return True
    return False

def play_game(agent, opponent, episode, save_png=False, save_path='game_visualizations'):
    board = np.zeros((3, 3), dtype=int)
    turn = 1
    total_reward = 0

    while True:
        if turn == 1:
            state = agent.get_state_key(board)
            action = agent.choose_action(board)
            agent.update_q_values(state, action, 0, board)
            board[action] = 1
        else:
            action = opponent(board)
            board[action] = -1

        if check_winner(board):
            total_reward = 1 if turn == 1 else 0
            break
        elif np.all(board != 0):
            total_reward = 0.5
            break

        turn = 3 - turn

    if save_png and episode % 150 == 0:
        save_path = os.path.join(save_path, f'game_{episode}.png')
        save_game_visualization(board, save_path)

    return total_reward

def save_game_visualization(board, save_path):
    cmap = plt.cm.colors.ListedColormap(['white', 'green', 'yellow'])
    bounds = [-1, 0, 1, 2]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(board, cmap=cmap, norm=norm, interpolation='none', origin='upper', extent=[0, 3, 0, 3])
    plt.xticks(np.arange(0, 4, 1))
    plt.yticks(np.arange(0, 4, 1))
    plt.grid(color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.savefig(save_path)
    plt.close()

def train_agent(agent, episodes=1000, save_visualizations=True):
    wins = 0
    results = []

    for episode in range(episodes):
        if episode % 150 == 0:
            results.append(wins)
            wins = 0

        agent_epsilon = max(0.1, agent.epsilon * (episodes - episode) / episodes)
        agent.epsilon = agent_epsilon

        if episode % 100 == 0:
            print(f'Episode {episode}, epsilon: {agent_epsilon}')

        result = play_game(agent, opponent=random_opponent, episode=episode, save_png=save_visualizations)

        if result == 1:
            wins += 1

    agent.save_q_values('q_values.pkl')

    plt.plot(results)
    plt.xlabel('Episodes (x 150)')
    plt.ylabel('Wins')
    plt.show()

def random_opponent(board):
    available_actions = [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]
    return random.choice(available_actions)

if __name__ == "__main__":
    agent = TicTacToeAgent()
    train_agent(agent, episodes=10000, save_visualizations=True)
