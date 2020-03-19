import numpy as np
import numpy.random as rg
import matplotlib.pyplot as plt
import tensorflow as tf
from gym_cube_v2 import Cube, nFace, maxTurn, ele_size

def append_sample(state, action, reward, next_state, done):
    memory.append([state.flatten(), action, reward, next_state.flatten(), done])

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    sum_exp_logits = np.sum(exp_logits)
    return exp_logits / sum_exp_logits

def action2index(action):
    return (action + nFace) if action < 0 else (action + nFace - 1)

def index2action(index):
    return (index - nFace) if index < nFace else (index - nFace + 1)

def action2onehot(action):
    return np.eye(nFace*2)[action2index(action)]

def nturn2onehot(nturn):
    return np.eye(maxTurn)[nturn - 1]

def onehot2nturn(onehot):
    return np.argmax(onehot) + 1 if np.sum(onehot) == 1.0 else -1

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=5000, activation='relu', input_shape=(ele_size,), kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(units=1000, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(units=1000, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(units=1000, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(units=1, activation='relu')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')
    model.summary()
    return model

env = Cube()
actions = env.action_space.actions
action_size = env.action_space.size()

batch_size = 128

model = build_model()
model_e = build_model()
model_e.set_weights(model.get_weights())

def evaluate_model():
    nTurns = rg.randint(1, maxTurn+1)

    env.reset()
    env.shuffle(nTurn)
    env.render()

    iter = 0
    while iter < 200:
        ys = np.zeros(action_size)
        for j in range(action_size):
            obv, reward, done, info = env.step(actions[j])
            ys[j] = model.predict(np.expand_dims(obv, axis=0))
            env.step(-actions[j])

        best_action = actions[argmin(ys)]
        env.step(best_action)
        env.render()
        iter += 1

M = 100
C = 10
epsilon = 1.5

for m in range(M):
    nTurns = sorted(rg.randint(1, maxTurn+1, size=batch_size))
    # print(nTurns)

    X = np.zeros((batch_size, ele_size))
    yh = np.zeros(batch_size)

    for i in range(batch_size):
        env.reset()
        nTurn = nTurns[i]
        X[i] = env.shuffle(nTurn)
        # env.render()

        ys = np.zeros(action_size)
        for j in range(action_size):
            obv, reward, done, info = env.step(actions[j])
            # env.render()
            # print(done)
            p = 0. if done else model_e.predict(np.expand_dims(obv, axis=0))
            ys[j] = 1 + p
            env.step(-actions[j])

        yh[i] = np.min(ys)

    # print(yh)

    history = model.fit(X, yh, batch_size=batch_size, epochs=1, verbose=1)

    loss = history.history['loss'][-1]


    if m % C == 0 and loss < epsilon:
        model_e.set_weights(model.get_weights())
        print('loss = %.3f' % loss)

evaluate_model()
