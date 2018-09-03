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


from config import *

df = pd.read_csv('../train.csv')
train_files = [f for f in df.images.values]
file_id_mapping =  {k: v for k, v in zip(df.images.values, df.classes.values)}

with open('feature_3.pkl', 'r') as f:
    train_preds, test_preds, test_file_names = pickle.load(f)


neigh = NearestNeighbors(n_neighbors=22)
neigh.fit(train_preds)


distances_test, neighbors_test = neigh.kneighbors(test_preds)
distances_test, neighbors_test = distances_test.tolist(), neighbors_test.tolist()

def pred():
    preds_str = []

    with open('submission.csv', 'w') as f:
        f.write('filename|defect,probability\n')
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

            labels, scores = zip(*sample_result)
            print ' '.join(labels), scores
            labels = [defect_dict[l] if l in defect_dict else 'defect_10' for l in labels]

            ps = [np.clip(labels.count(d) / 22.0, 0.001, 0.999) for d in defect_list]
            # print ps
            ps = ps / np.sum(ps)
            # print ps
            for d, p in zip(defect_list, ps):
                f.write('{}|{}, {}\n'.format(filepath,d,p))

            
pred()