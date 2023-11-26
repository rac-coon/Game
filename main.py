import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.algorithms.ppo import PPO
import numpy as np
import random
import os
import matplotlib.pyplot as plt

ray.init()

def opponent_logic(observation):
    for row in observation:
        if row.count(1) == 2 and row.count(0) == 0:
            return row.index(-1)
    return random.choice([i for i, x in enumerate(observation[0]) if x == -1])

class TicTacToeEnv:
    def __init__(self, _):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for agent, -1 for opponent
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            done = self.check_winner() or self.is_board_full()
            reward = 1 if done and self.current_player == 1 else 0
            self.current_player *= -1  # Switch player
            return self.board.flatten(), reward, done, {}
        else:
            # Invalid move, penalize the agent
            return self.board.flatten(), -1, True, {}

    def check_winner(self):
        for i in range(3):
            if all(self.board[i, :] == 1) or all(self.board[:, i] == 1):
                return True
            if all(self.board[i, :] == -1) or all(self.board[:, i] == -1):
                return True
        if all(np.diag(self.board) == 1) or all(np.diag(np.fliplr(self.board)) == 1):
            return True
        if all(np.diag(self.board) == -1) or all(np.diag(np.fliplr(self.board)) == -1):
            return True
        return False

    def is_board_full(self):
        return not any(0 in row for row in self.board)

class CustomModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.fcnet = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.fcnet.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.fcnet.value_function()

ModelCatalog.register_custom_model("custom_model", CustomModel)

config = {
    "env": TicTacToeEnv,
    "model": {"custom_model": {}},
    "framework": "tf",
    "num_workers": 1,
    "env_config": {},
}

analysis = PPO.PPOTrainer(config=config)
knowledge_base_path = "knowledge_base.npy"
stats_path = "training_stats.txt"
graph_path = "training_graph.png"
knowledge_base = np.load(knowledge_base_path) if os.path.exists(knowledge_base_path) else np.zeros((3, 3))
training_stats = {"wins": 0, "losses": 0}
graph_data = {"episode": [], "wins": []}

def update_knowledge_base(observation, reward):
    global knowledge_base
    knowledge_base += (observation.reshape(3, 3) == 0) * reward

def save_knowledge_base():
    np.save(knowledge_base_path, knowledge_base)

def save_stats():
    with open(stats_path, "w") as file:
        file.write(f"Wins: {training_stats['wins']}\n")
        file.write(f"Losses: {training_stats['losses']}\n")

def update_stats(reward):
    global training_stats
    if reward == 1:
        training_stats["wins"] += 1
    else:
        training_stats["losses"] += 1

def save_graph():
    plt.plot(graph_data["episode"], graph_data["wins"])
    plt.xlabel("Episode")
    plt.ylabel("Wins")
    plt.title("Training Progress")
    plt.savefig(graph_path)
    plt.show()

for episode in range(1, 1001):
    result = analysis.train()
    episode_reward = result["episode_reward_mean"]
    update_stats(episode_reward)
    update_knowledge_base(analysis.get_policy().model.base_env.observation_space.sample(), episode_reward)

    if episode % 150 == 0:
        graph_data["episode"].append(episode)
        graph_data["wins"].append(training_stats["wins"])
        save_graph()

save_knowledge_base()
save_stats()

ray.shutdown()
