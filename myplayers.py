from catanatron import Player
from catanatron_experimental.cli.cli_players import register_player
import random
from catanatron.models.actions import ActionType

WEIGHTS_BY_ACTION_TYPE = {
    ActionType.BUILD_CITY: 10000,
    ActionType.BUILD_SETTLEMENT: 1000,
    ActionType.BUY_DEVELOPMENT_CARD: 100,
}

@register_player("FOO")
class FooPlayer(Player):
  def decide(self, game, playable_actions):
        bloated_actions = []
        for action in playable_actions:
            print(action.action_type)
            weight = WEIGHTS_BY_ACTION_TYPE.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        index = random.randrange(0, len(bloated_actions))
        return bloated_actions[index]
    #return playable_actions[0]
    # ===== END YOUR CODE =====
