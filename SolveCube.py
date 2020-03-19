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
        tf.keras.layers.Dense(units=26, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy')
    model.summary()
    return model

env = Cube()

batch_size = 512
iter_max = 10000
model = build_model()

# def train_model():
for k in range(iter_max):
    targets = rg.randint(1, maxTurn+1, size=batch_size)
    states = np.zeros((batch_size, ele_size))

    for i in range(batch_size):
        nTurn = targets[i]
        env.reset()
        env.shuffle(nTurn)

        states[i] = env.observation_space.onehot()

    model.fit(states, nturn2onehot(targets), batch_size=batch_size, epochs=1, verbose=1)

# def evaluate_model():
targets = rg.randint(1, maxTurn+1, size=20)

for nTurn in targets:
    env.reset()
    env.shuffle(nTurn)

    state = env.observation_space.onehot()

    nTurn_hat = model.predict(np.expand_dims(state, axis=0))

    print(nTurn, onehot2nturn(nTurn_hat))
