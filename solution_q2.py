import gymnasium as gym
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

class transFunc():

    def __init__(self, startstate: int, action: int, endstate: int) -> None:
        self.sstate = startstate
        self.estate = endstate
        self.action = action
        self.estate = {}
        self.estate[endstate] = 1
        self.samplesize = 1
        # self.next = None

    # adds a single observation of ending on a state
    def add(self, endstate: int) -> None:
        if self.estate.get(endstate):
            self.estate[endstate] += 1
        else:
            self.estate[endstate] = 1
        self.samplesize += 1

    def getProb(self, endstate: int) -> float:
        return self.estate.get(endstate) / self.samplesize

    def print(self) -> int:
        node = self
        count = 0
        for endstate in self.estate.keys():
            print("start: {},  action: {},  end: {},  n: {},  chance: {}\n".format(node.sstate, node.action,
                                                                            endstate, node.samplesize, node.getProb(endstate)))
            if endstate == 15:
                count += 1
        return count
    


# Q2.2
def learn(Rarray: []) -> {}:
    observation, info = env.reset()
    startState = observation
    out = {}

    def tokey(inital: int, action: int) -> int:
        return (inital << 4) + action

    for x in range(1000):
        terminated = 0
        truncated = 0
        while not terminated and not truncated:
            action = env.action_space.sample() # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            # record to learning table
            key = tokey(startState, action)
            if out.get(key):
                out[key].add(observation)
            else:
                out[key] = transFunc(startState, action, observation)
                Rarray[observation] = reward
            
            startState = observation

        observation, info = env.reset()
        startState = observation

    return out


if __name__ == "__main__":
    Rarray = [0.0 for x in range(16)]
    Tarray = learn(Rarray)

    # for x in sorted(Tarray.keys()):
    #     Tarray[x].print()
    # for x in Rarray:
    #     print("{} ".format(x))

    env.close()
    
