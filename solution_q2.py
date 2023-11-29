import time

import gymnasium as gym

class TFunc():

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
# Note: possible crash, 1000 is not enough to guarantee an action that reaches the goal
def learn(Rarray: [], env) -> {}:
    observation, info = env.reset()
    startState = observation
    out = {}

    for _ in range(1000):
        terminated = 0
        truncated = 0
        while not terminated and not truncated:
            action = env.action_space.sample() # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            # record to learning table
            key = tokey(startState, action)
            if out.get(key):
                out[key].add(observation)
                Rarray[observation] = max(reward, Rarray[observation])
            else:
                out[key] = TFunc(startState, action, observation)
                Rarray[observation] = reward
            
            startState = observation

        observation, info = env.reset()
        startState = observation

    return out

discount_factor = 1.0
# Q2.3
def study(T: {}, R: []) -> []:
    V = R
    # iteration layer
    for i in range(100):
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

# Q2.4
def solve(T: {}, R: [], V: []) -> []:
    P = [0 for x in range(16)]
    for s in range(16):
        m = 0.0
        ma = -1
        # for each action
        for a in range(4):
            if func := T.get(tokey(s, a)):
                # summation
                sum = 0
                for x in func.getEndState():
                    sum += func.getProb(x) * (R[s] + discount_factor * V[x])
                if sum > m:
                    m = sum
                    ma = a
        P[s] = ma
    # printGrid(P)
    return P

# Q2.5
def apply(P: [], state: int) -> int:
    out = P[state]
    if out < 0:
        raise Exception("Can not provide action for terminal state")
    return out

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    Rarray = [0.0 for x in range(16)]
    Tarray = learn(Rarray, env)
    env.close()

    # for x in sorted(Tarray.keys()):
    #     Tarray[x].print()
    # for x in Rarray:
    #     print("{} ".format(x))

    Varray = study(Tarray, Rarray)
    ActionArray = solve(Tarray, Rarray, Varray)

    # testing policy
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode="human",
                    is_slippery=True, )

    terminated = 0
    truncated = 0
    observation, info = env.reset()
    while not terminated and not truncated:
        # if this crashes, see note on Q2.2
        action = apply(ActionArray, observation)
        observation, reward, terminated, truncated, info = env.step(action)

    env.close()
    
