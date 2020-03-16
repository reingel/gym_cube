import numpy as np
import numpy.random as rg
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from gym_cube import Gym_cube

env = Gym_cube()

max_episodes = env._max_episode_steps
scores = []
steps = []
iteration = 0

nFace = env.nFace
nDot = env.nDot
ele_size = nFace * nDot * nDot
layer_count = nFace

epsilon = 0.9
epsilon_min = 0.01

gamma = 0.9
batch_size = 512
max_memory = batch_size * 8
memory = []
train_count = 0

def append_sample(state, action, reward, next_state, done):
    memory.append([state, action, reward, next_state, done])

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    sum_exp_logits = np.sum(exp_logits)
    return exp_logits / sum_exp_logits

def build_model():
    dense1 = 128
    dense2 = 128

    x = tf.keras.Input(shape=(ele_size,))
    d1 = tf.keras.layers.Dense(dense1, activation='relu')(x)
    d2 = tf.keras.layers.Dense(dense2, activation='relu')(d1)
    out = tf.keras.layers.Dense(nFace*2, activation='sigmoid')(d2)

    model = tf.keras.Model(inputs=x, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005), loss='mse')
    model.summary()
    return model

def train_model():
    np.random.shuffle(memory)

    len = max_memory // batch_size
    for k in range(len):
        mini_batch = memory[k*batch_size:(k+1)*batch_size]

        states = np.zeros((batch_size, ele_size))
        next_states = np.zeros((batch_size, ele_size))
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = model.predict(states)
        next_target = target_model.predict(next_states)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + gamma * np.amax(next_target[i])

        model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)


model = build_model()
target_model = build_model()

for i in range(max_episodes):
    if i % 100 == 0 and i != 0:
        print(i, 'mean score: %.3f, mean step: %d, iteration: %d, epsilon: %.3f' %
              (np.mean(scores[-100:]), np.mean(steps[-100:]), iteration, epsilon))

    prev_obs = env.reset()
    print(prev_obs)
    score = 0
    step = 0

    while True:
        if rg.random() < epsilon:
            action = env.action_space.sample()
        else:
            x = prev_obs
            logits = model.predict(x.ravel())
            prob = softmax(logits)
            action = np.argmax(prob)

        obs, reward, done, info = env.step(action)

        score += reward
        step += 1

        append_sample(prev_obs, action, reward, obs, done)

        if len(memory) >= max_memory:
            train_model()
            memory = []

            train_count += 1
            if train_count % 4 == 0:
                target_model.set_weights(model.get_weights())

        prev_obs = obs

        if epsilon > epsilon_min and iteration % 250 == 0:
            epsilon = epsilon * gamma

        if done:
            break

    scores.append(score)
    steps.append(step)
