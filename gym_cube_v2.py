import numpy as np

nFace = 6 # no. of faces in a cube
nDot = 3 # no. of dots in an edge
ele_size = (nDot**2 * nFace) * nFace
maxTurn = 26 # theoretical max. of turns to solve

class Observation_space:
    def __init__(self, isVerbose=False):
        self.states = np.arange(1, nFace+1, dtype=int).repeat(nDot**2).reshape(nFace,nDot,nDot)
        if isVerbose:
            for i in range(nFace):
                for j in range(nDot):
                    for k in range(nDot):
                        self.states[i,j,k] = (i+1)*100 + j*10 + k

    def onehot(self):
        return np.eye(nFace)[self.states - 1].flatten()

class Action_space:
    def __init__(self):
        self.actions = np.hstack((np.arange(-nFace, 0), np.arange(1, nFace + 1)))

    def size(self):
        return 2*nFace

    def sample(self):
        return self.actions[np.random.randint(0, self.actions.size)]

class Cube:
    def __init__(self, isVerbose=False):
        self.observation_space = Observation_space(isVerbose)
        self.action_space = Action_space()

        self._step = 0
        self._max_episode_steps = 26*100

        # internal properties
        # rotation matrix
        self.A = np.zeros((nDot * 3, nDot * 3))

        # coord of rotation matrix for top, left, center, right, bottom
        self.row = np.array([0, 1, 1, 1, 2]) * nDot
        self.col = np.array([1, 0, 1, 2, 1]) * nDot

        # id of faces: top, left, center, right, bottom
        self.idFace = np.array([[4, 5, 1, 3, 2],       # 1
                                [1, 5, 2, 3, 6],       # 2
                                [1, 2, 3, 4, 6],       # 3
                                [1, 3, 4, 5, 6],       # 4
                                [1, 4, 5, 2, 6],       # 5
                                [2, 5, 6, 3, 4]]) - 1  # 6
        # orientation of faces: top, left, center, right, bottom
        self.orFace = np.array([[2, 3, 0, 1, 0],       # 1
                                [0, 0, 0, 0, 0],       # 2
                                [3, 0, 0, 0, 1],       # 3
                                [2, 0, 0, 0, 2],       # 4
                                [1, 0, 0, 0, 3],       # 5
                                [0, 1, 0, 3, 2]])      # 6

        self.symbol = []
        for i in range(nFace):
            self.symbol.append('\x1b[0;97;%sm %s \x1b[0m' % (str(40 + i), str(i + 1)))

    #
    # internals
    #

    def rotate(self, cmd):
        n = nDot
        c = np.abs(cmd) - 1
        d = np.sign(cmd)
        idx = self.idFace[c]
        rot = self.orFace[c]
        row = self.row
        col = self.col

        for i in range(5):
            self.A[row[i]:row[i]+n, col[i]:col[i]+n] = np.rot90(self.observation_space.states[idx[i]], rot[i])
        self.A[n-1:2*n+1, n-1:2*n+1] = np.rot90(self.A[n-1:2*n+1, n-1:2*n+1], d)
        for i in range(5):
            self.observation_space.states[idx[i]] = np.rot90(self.A[row[i]:row[i]+n, col[i]:col[i]+n], -rot[i])

    def shuffle(self, n=30):
        for i in range(n):
            action = self.action_space.sample()
            self.rotate(action)

        observation = self.observation_space.onehot()

        return observation

    def isSolved(self):
        return all(np.unique(face).size == 1 for face in self.observation_space.states)

    def evaluate(self):
        return 0. if self.isSolved() else 1.

    def symbolize(self, state):
        ret = ''
        for i in state:
            ret += self.symbol[i-1]
        return ret

    #
    # interfaces
    #

    def reset(self):
        self._step = 0
        self.observation_space.__init__()
        observation = self.observation_space.states
        return observation

    def render(self):
        print()
        for i in range(nDot):
            print(self.symbolize(self.observation_space.states[0][i]))
        for i in range(nDot):
            print(self.symbolize(self.observation_space.states[1][i]),
                  self.symbolize(self.observation_space.states[2][i]),
                  self.symbolize(self.observation_space.states[3][i]),
                  self.symbolize(self.observation_space.states[4][i]))
        for i in range(nDot):
            print(self.symbolize(self.observation_space.states[5][i]))
        print()

    def step(self, action):
        self.rotate(action)
        self._step += 1

        observation = self.observation_space.onehot()
        reward = self.evaluate()
        done = self.isSolved() or (self._step >= self._max_episode_steps)
        info = None

        return observation, reward, done, info

    def close(self):
        pass


if __name__ == '__main__':
    env = Cube()
    env.reset()

    iter = 0
    score = 0.
    while True:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)

        if iter % 1000 == 0:
            print('%d, %.3f' % (iter, score))
            env.render()

        iter += 1
        score += reward

        if done or iter > 10000:
            break
