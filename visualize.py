import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

save_dir = Path('./data/xor_2')

with open(save_dir / 'wave_height.pkl', mode='rb') as f:
    wave_height = pickle.load(f)

# preprocess
wave_height = wave_height - np.mean(wave_height)
wave = wave_height.reshape(len(wave_height), -1)

fig, ax = plt.subplots()

ims = []
for i in range(len(wave_height)-10):
    im = ax.imshow(wave_height[i+10], animated=True, vmin=-0.5, vmax=0.5)
    if i == 0:
        ax.imshow(wave_height[i+10], vmin=-0.5, vmax=0.5)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)

plt.show()

