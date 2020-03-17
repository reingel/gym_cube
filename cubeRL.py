import numpy as np
import numpy.random as rg
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from gym_cube import Gym_cube

env = Gym_cube()

max_episodes = 1000
scores = []
steps = []
iteration = 0

nFace = env.nFace
nDot = env.nDot
ele_size = nFace * nDot * nDot
layer_count = nFace

epsilon = 0.9
epsilon_min = 0.01
decay = 0.99

gamma = 0.99
batch_size = 512
batch_len = 8
max_memory = batch_size * batch_len
memory = []
train_count = 0

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
    dense1 = 128
    dense2 = 128

    x = tf.keras.Input(shape=(ele_size,))
    d1 = tf.keras.layers.Dense(dense1, activation='relu')(x)
    d2 = tf.keras.layers.Dense(dense2, activation='relu')(d1)
    y = tf.keras.layers.Dense(nFace*2, activation='sigmoid')(d2)

    model = tf.keras.Model(inputs=x, outputs=y)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005), loss='mse')
    model.summary()
    return model

def train_model():
    np.random.shuffle(memory)

    for k in range(batch_len):
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
                target[i][action2index(actions[i])] = rewards[i]
            else:
                target[i][action2index(actions[i])] = rewards[i] + gamma * np.amax(next_target[i])

        model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)


model = build_model()
target_model = build_model()

for i in range(max_episodes):
    if i % 10 == 0 and i != 0:
        print(i, 'mean score: %.3f, mean step: %d, iteration: %d, epsilon: %.3f' %
              (np.mean(scores[-10:]), np.mean(steps[-10:]), iteration, epsilon))

    prev_obs = env.reset()
    prev_action = env.action_space.sample()
    pprev_action = env.action_space.sample()
    score = 0
    step = 0

    while True:
        if rg.random() < epsilon:
            action = env.action_space.sample()
            if action == prev_action and action == pprev_action:
                action = env.action_space.sample()
        else:
            x = prev_obs
            logits = model.predict(np.expand_dims(x.flatten(), axis=0))
            prob = softmax(logits)
            index = np.argmax(prob)
            action = index2action(index)

        obs, reward, done, info = env.step(action)

        score += reward
        step += 1

        if step % 1000 == 0:
            if len(memory) >= 4:
                act = [memory[-4][1], memory[-3][1], memory[-2][1], memory[-1][1]]
                print('%d, %d, %.3f, %s' % (train_count, step, score, str(act)))
                env.render()

        append_sample(prev_obs, action, reward, obs, done)

        if len(memory) >= max_memory:
            train_model()
            memory = []

            train_count += 1
            if train_count % 4 == 0:
                target_model.set_weights(model.get_weights())

        prev_obs = obs
        pprev_action = prev_action
        prev_action = action
        iteration += 1

        if done:
            break

    if epsilon > epsilon_min:
        epsilon = epsilon * decay

    scores.append(score)
    steps.append(step)
