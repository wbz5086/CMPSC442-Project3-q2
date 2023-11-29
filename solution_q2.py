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

    def getEndState(self) -> []:
        return self.estate.keys()

    def print(self) -> int:
        node = self
        count = 0
        for endstate in self.estate.keys():
            print("start: {},  action: {},  end: {},  n: {},  chance: {}\n".format(node.sstate, node.action,
                                                                            endstate, node.samplesize, node.getProb(endstate)))
            if endstate == 15:
                count += 1
        return count
    
def tokey(inital: int, action: int) -> int:
    return (inital << 4) + action

# prints a square grid of size 4 by 4, given a size 16 array
def printGrid(grid: []):
    out = ""
    for x in range(16):
        if not x % 4:
            out += "\n"
        out += "{:0.2f} ".format(grid[x])
    print(out)

# Q2.2
def learn(Rarray: []) -> {}:
    observation, info = env.reset()
    startState = observation
    out = {}

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

discount_factor = 0.9
# Q2.3
def study(T: {}, R: []) -> []:
    V = R
    # iteration layer
    for i in range(50):
        V1 = [0.0 for x in range(16)]
        # for each state
        for s in range(16):
            m = 0.0
            # for each action
            for a in range(4):
                if func := T.get(tokey(s, a)):
                    # summation
                    sum = 0
                    for x in func.getEndState():
                        sum += func.getProb(x) * (R[s] + discount_factor * V[x])
                    m = max(sum, m)
            if m == 0.0:
                m = R[s]
            V1[s] = m
        V = V1
        # printGrid(V)
    return V


def solve():
    pass

def apply():
    pass

if __name__ == "__main__":
    Rarray = [0.0 for x in range(16)]
    Tarray = learn(Rarray)

    # for x in sorted(Tarray.keys()):
    #     Tarray[x].print()
    # for x in Rarray:
    #     print("{} ".format(x))

    Varray = study(Tarray, Rarray)

    env.close()
    
