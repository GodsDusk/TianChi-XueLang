import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
import pickle
import numpy as np
import glob
from tqdm import tqdm

from model_utils import build_inference_model
from data_utils import data_generator
inference_model = build_inference_model(weight_path='vgg_all.h5')

train_files = glob.glob("../data/xuelang_round1_train_*/*/*.jpg")
test_files = glob.glob("../data/xuelang_round1_test_b/*.jpg")


def get_feature(files, batch_size=8):

    preds = []
    file_names = []
    for fnames, imgs in tqdm(data_generator(files, batch=batch_size), total=len(files)/batch_size):

        predicts = inference_model.predict(imgs)
        preds += predicts.tolist()
        file_names += fnames
        
    return np.array(preds), file_names





train_preds, train_file_names = get_feature(train_files)
test_preds, test_file_names = get_feature(test_files)


with open('feature.pkl', 'w') as f:
    pickle.dump([train_preds, test_preds, test_file_names], f)
