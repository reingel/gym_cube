import numpy as np

class Cubelet:
    def __init__(self, isVerbose=False):



class Observation_space:
    def __init__(self, nDot, nFace, isVerbose=False):
        self.states = np.arange(1, nFace+1, dtype=int).repeat(nDot**2).reshape(nFace,nDot,nDot)
        if isVerbose:
            for i in range(nFace):
                for j in range(nDot):
                    for k in range(nDot):
                        self.states[i,j,k] = (i+1)*100 + j*10 + k

class Action_space:
    def __init__(self, nFace=6):
        self.actions = np.hstack((np.arange(-nFace, 0), np.arange(1, nFace + 1)))

    def sample(self):
        return self.actions[np.random.randint(0, self.actions.size)]

class Cube:
    def __init__(self, isVerbose=False):
        self.nDot = 3 # no. of dots in an edge
        self.nFace = 6 # no. of faces in a cube
        self.observation_space = Observation_space(self.nDot, self.nFace, isVerbose)
        self.action_space = Action_space(self.nFace)

        self._step = 0
        self._max_episode_steps = 26*100

        # internal properties
        # rotation matrix
        self.A = np.zeros((self.nDot * 3, self.nDot * 3))

        # coord of rotation matrix for top, left, center, right, bottom
        self.row = np.array([0, 1, 1, 1, 2]) * self.nDot
        self.col = np.array([1, 0, 1, 2, 1]) * self.nDot

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
        for i in range(self.nFace):
            self.symbol.append('\x1b[0;97;%sm %s \x1b[0m' % (str(40 + i), str(i + 1)))

        self.total = 0
        self.prev_total = 0

    def reset(self):
        self._step = 0
        self.shuffle()
        observation = self.observation_space.states
        return observation

    def rotate(self, cmd):
        if not (-self.nFace <= cmd and cmd <= self.nFace and cmd != 0):
            return None

        n = self.nDot
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
            action = np.random.randint(-self.nFace, self.nFace + 1)
            self.rotate(action)

    def evaluate(self):
        self.total = np.sum(np.max(face) - np.min(face) for face in self.observation_space.states)
        ret = 100. if self.total == 0 else 0.01 if self.total > self.prev_total else 0.

        self.prev_total = self.total

        return ret

    def isSolved(self):
        return all(np.min(face) == np.max(face) for face in self.observation_space.states)

    def symbolize(self, state):
        ret = ''
        for i in state:
            ret += self.symbol[i-1]
        return ret

    def render(self):
        for i in range(self.nDot):
            print(self.symbolize(self.observation_space.states[0][i]))
        for i in range(self.nDot):
            print(self.symbolize(self.observation_space.states[1][i]),
                  self.symbolize(self.observation_space.states[2][i]),
                  self.symbolize(self.observation_space.states[3][i]),
                  self.symbolize(self.observation_space.states[4][i]))
        for i in range(self.nDot):
            print(self.symbolize(self.observation_space.states[5][i]))
        print()

    def step(self, action):
        self.rotate(action)
        self._step += 1

        observation = self.observation_space.states
        reward = self.evaluate()
        done = self.isSolved() or (self._step >= self._max_episode_steps)
        info = None

        return observation, reward, done, info

    def close(self):
        pass


if __name__ == '__main__':
    env = Gym_cube()
    env.reset()

    iter = 0
    score = 0.
    while True:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)

        if iter % 10000 == 0:
            print('%d, %.3f' % (iter, score))
            env.render()

        iter += 1
        score += reward

        if done:
            break
