import glob
import os
import pickle

import numpy as np
from utils import get_dataset
import tensorflow as tf
import matplotlib.pyplot as plt

paths = glob.glob('data/waymo/training_and_validation/*')
paths.sort()

cdistribution = []
#counts[,vehicle, pedestrian, cyclist]
counts = np.zeros((len(paths), 3))
#Datadump for offline EDA
data = []

for i, path in enumerate(paths):

    print('Proccesing ',path)
    dataset = get_dataset(path)
    dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
    batch = dataset.take(10)

    for j, rec in enumerate(batch):
        data.append(rec)
        classes = rec['groundtruth_classes'].numpy()
        _unique, count = np.unique(classes, return_counts=True)
        for k in range(len(count)):
            counts[i, k] += count[k]     

labels = np.arange(0,len(counts))

width = 0.8

fig, ax = plt.subplots()

ax.bar(labels, counts[:, 0], width, label='Vehicle')
ax.bar(labels, counts[:, 1], width, label='Pedestrian')
ax.bar(labels, counts[:, 2], width, label='Cyclist')

ax.set_ylabel('Number of Occurences')
ax.set_xlabel('Number for the corresponding tfRecord-file')
ax.set_title('Occurences of Classes')
ax.legend()

plt.savefig('test.png')
