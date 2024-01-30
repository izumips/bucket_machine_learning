import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

def ridge(data, target, beta=0.001):
    N = data.shape[1]
    M = target.shape[1]

    data, target = data.T, target.T # match the dimension to the textbook

    W = target @ data.T @ np.linalg.inv(data @ data.T + beta * np.identity(N))

    return W    

with open('./data/xor_0/wave_height.pkl', mode='rb') as f:
    xor_0 = pickle.load(f)

with open('./data/xor_1/wave_height.pkl', mode='rb') as f:
    xor_1 = pickle.load(f)

with open('./data/xor_2/wave_height.pkl', mode='rb') as f:
    xor_2 = pickle.load(f)

with open('./data/xor_3/wave_height.pkl', mode='rb') as f:
    xor_3 = pickle.load(f)

xor_0 = xor_0.reshape(len(xor_0), -1) # 0
lab_0 = np.zeros((len(xor_0), 1))
tes_0 = np.array([[0, 0]] * len(xor_0))

xor_1 = xor_1.reshape(len(xor_1), -1) # 1
lab_1 = np.ones((len(xor_1), 1))
tes_1 = np.array([[0, 1]] * len(xor_1))

xor_2 = xor_2.reshape(len(xor_2), -1) # 1
lab_2 = np.ones((len(xor_2), 1))
tes_2 = np.array([[1, 0]] * len(xor_2))

xor_3 = xor_3.reshape(len(xor_3), -1) # 0
lab_3 = np.zeros((len(xor_3), 1))
tes_3 = np.array([[1, 1]] * len(xor_3))

data = np.vstack((xor_0, xor_1, xor_3, xor_2))
test = np.vstack((tes_0, tes_1, tes_3, tes_2))
label = np.vstack((lab_0, lab_1, lab_3, lab_2))

W = ridge(data, label)
W2 = ridge(test, label)

out = np.array([W @ d for d in data])
out2 = np.array([W2 @ t for t in test])

plt.figure(figsize=(10, 6))

plt.plot(label, color='k', linewidth=3, linestyle='dotted', label='training data')
plt.plot(out, color='tab:blue', alpha=0.8, linewidth=1, label='prediction from wave height')
plt.plot(out2, color='tab:orange', alpha=0.8, linewidth=2, label='prediction from original data')

plt.legend()
plt.show()