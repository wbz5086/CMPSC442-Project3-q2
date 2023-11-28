import gymnasium as gym
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

class transNode():

    def __init__(self, startstate: int, action: int, endstate: int, reward: float) -> None:
        self.sstate = startstate
        self.estate = endstate
        self.action = action
        self.reward = reward
        self.next = None

    def setNext(self, next) -> None:
        self.next = next

    def print(self) -> int:
        node = self
        count = 0
        while node is not None:
            print("start: {},  end: {},  reward: {}\n".format(node.sstate, node.estate, node.reward))
            if node.reward == 1:
                count += 1
            node = node.next
        return count


# Q2.2
def learn() -> []:
    observation, info = env.reset()
    startState = 0
    out = [None for p in range(1000)]
    for x in range(1000):
        terminated = 0
        truncated = 0
        node = transNode(-1, -1, startState, 0.0)
        out[x] = node
        while not terminated and not truncated:
            action = env.action_space.sample() # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            # record to learning table
            node.next = transNode(startState, action, observation, reward)
            node = node.next
            startState = observation

        observation, info = env.reset()
        node.next = transNode(startState, -1, -1, 0.0)
        startState = 0
        # print("Observation: {}   Reward: {}   info: {}\n".format(observation, reward, info))
    return out


if __name__ == "__main__":
    learnarray = learn()
    total = 0
    for x in learnarray:
        total += x.print()
    print(total)

    env.close()
    
