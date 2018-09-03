#coding=utf-8
import os
import sys
reload(sys) 
sys.setdefaultencoding("utf-8")

import pickle
import pandas as pd
import numpy as np
from glob import glob
from sklearn.neighbors import NearestNeighbors




train_files = glob('../data/xuelang_round1_train_*/*/*.jpg') 

file_id_mapping = {l:0 if u'正常' in l else 1 for l in train_files}

with open('feature.pkl', 'r') as f:
    train_preds, test_preds, test_file_names = pickle.load(f)



neigh = NearestNeighbors(n_neighbors=6)
neigh.fit(train_preds)


distances_test, neighbors_test = neigh.kneighbors(test_preds)
distances_test, neighbors_test = distances_test.tolist(), neighbors_test.tolist()

def pred():
    preds_str = []

    for filepath, distance, neighbour_ in zip(test_file_names, distances_test, neighbors_test):

        # print filepath, distance, neighbour_

        sample_result = []
        sample_classes = []

        for d, n in zip(distance, neighbour_):
            train_file = train_files[n]
            class_train = file_id_mapping[train_file]
            sample_classes.append(class_train)
            sample_result.append((class_train, d))
        # print sample_classes, sample_result
        

        sample_result.sort(key=lambda x: x[1])

        labels, scores = np.array(zip(*sample_result))

        # scores = scores / np.sum(scores)

        # pred = np.sum(np.multiply(labels, scores)) / np.sum(scores)

        pred = sum(labels) / 6.0

        # print labels, scores,  np.sum(np.multiply(labels, scores)) / np.sum(scores), np.sum(labels) / 6.0, np.sum(np.multiply(labels, scores)) / np.sum(scores) - np.sum(labels) / 6.0

        preds_str.append(np.clip(pred, 0.0001, 0.99999))
    return preds_str

preds_str = pred()

df = pd.DataFrame([x.split(os.sep)[-1] for x in test_file_names], columns=["filename"])
df['probability'] = preds_str
df.to_csv("submission.csv", index=False)
