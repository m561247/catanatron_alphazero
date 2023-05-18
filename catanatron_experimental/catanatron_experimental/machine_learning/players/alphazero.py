import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mcts import MCTSPlayer, StateNode
from catanatron.models.enums import Action, ActionPrompt, ActionType
from catanatron.models.player import Player, RandomPlayer, Color

from catanatron.game import Game
#from catanatron.models.player import Player
SIMULATIONS = 10

class AlphaZeroNetwork(nn.Module):
    def __init__(self, input_dim, action_space_size):
        super(AlphaZeroNetwork, self).__init__()
        print("before:",input_dim)
        self.fc1 = nn.Linear(input_dim, 100)  # input_dim should be 75 in your case
        
        self.fc_layers = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(256, action_space_size),
            nn.Softmax(dim=1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc_layers(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value



class AlphaZeroPlayer(MCTSPlayer):
    def __init__(self, color, network, num_simulations=SIMULATIONS, prunning=False):
        super().__init__(color, num_simulations, prunning)
        self.network = network
        #self.optimizer = optim.Adam(self.network.parameters())
        
    def decide(self, game: Game, playable_actions):
        # Modify the MCTS process to use the neural network's output
        # ...
        """
        root = StateNode(self.color, game.copy(), None, self.prunning)
        for _ in range(self.num_simulations):
            root.run_simulation()
        return root.choose_best_action()
        """
        return playable_actions[0]

    def self_play(self, num_games=1):
        # Implement the self-play process to generate training data
        training_data = []

        for _ in range(num_games):
            players = [MCTSPlayer(Color.RED),MCTSPlayer(Color.BLUE),MCTSPlayer(Color.WHITE)]
            game = Game(players)
            while not is_terminal():
                playable_actions = game.state.playable_actions
                # use MCTS policy
                best_action = self.decide(game, playable_actions)
                # to get the state and action
                game_state = game.state.player_state
                numpy_game_state = game_state_to_numpy(game_state)
                numpy_game_state = numpy_game_state.astype(float)  # convert to float
                numpy_game_state = np.expand_dims(numpy_game_state, axis=0)  # add batch dimension
                tensor_game_state = torch.from_numpy(numpy_game_state).float()  # convert to tensor
                policy, value = self.network(tensor_game_state)
                training_data.append((game_state, policy, value))
                
                # action
                game.execute(best_action)
            

        return training_data
        
    def train_network(self, training_data):
        # Train the neural network using the generated training data
        # ...
        pass


def game_state_to_numpy(game_state):
    # 提取游戏状态字典的值，并将它们转换为一个 NumPy 数组
    state_values = np.array(list(game_state.values()), dtype=np.float32)
    return state_values

def train_alpha_zero(iterations, num_simulations, prunning):
    
    network = AlphaZeroNetwork(input_dim=75, action_space_size=54)

    alpha_zero_player = AlphaZeroPlayer("red", network, num_simulations, prunning)

    for iteration in range(iterations):
        training_data = alpha_zero_player.self_play()
        print(training_data)
        alpha_zero_player.train_network(training_data)
    torch.save(network.state_dict(), "model_weights.pth")


def main():
    iterations = 100  
    num_simulations = 50  
    prunning = False

    train_alpha_zero(iterations, num_simulations, prunning)


if __name__ == "__main__":
    main()
