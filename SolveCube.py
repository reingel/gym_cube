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

def build_model():
    x = tf.keras.Input(shape=(ele_size,))
    d1 = tf.keras.layers.Dense(5000, activation='relu')(x)
    d2 = tf.keras.layers.Dense(1000, activation='relu')(d1)
    y = tf.keras.layers.Dense(nFace*2, activation='sigmoid')(d2)

    model = tf.keras.Model(inputs=x, outputs=y)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')
    model.summary()
    return model

env = Cube()

batch_size = 512
iter_max = 100
model = build_model()
model_e = build_model()

def train_model():
    for k in range(iter_max):
        targets = np.zeros(batch_size)
        states = np.zeros((batch_size, ele_size))
        next_states = np.zeros((batch_size, ele_size))
        actions = np.zeros(batch_size)
        rewards = np.zeros(batch_size)
        dones = np.zeros(batch_size)

        for i in range(batch_size):
            env.reset()
            nTurn = rg.randint(1, maxTurn+1)
            env.shuffle(nTurn)

            targets[i] = nTurn
            states[i] = env.observation_space.onehot()


            for action in env.action_space.actions:
                next_state, reward, done, _ = env.step(action)


        target = model.predict(states)
        next_target = target_model.predict(next_states)

        for i in range(batch_size):
            if dones[i]:
                target[i][action2index(actions[i])] = rewards[i]
            else:
                target[i][action2index(actions[i])] = rewards[i] + gamma * np.amax(next_target[i])

        model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)

