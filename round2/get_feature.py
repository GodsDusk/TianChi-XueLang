import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import pickle
import numpy as np
import glob
import pandas as pd 
from tqdm import tqdm

from model_utils import build_inference_model
from data_utils import data_generator

for i in xrange(1,8):
    inference_model = build_inference_model('hard_{}.h5'.format(i))

    # train_files = glob.glob("../data/xuelang_round1_train_*/*/*.jpg")

    df = pd.read_csv('../train.csv')
    train_files = [f for f in df.images.values]
    test_files = glob.glob("../data/xuelang_round2_test_b*/*.jpg")

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


    with open('feature_{}.pkl'.format(i), 'w') as f:
        pickle.dump([train_preds, test_preds, test_file_names], f)
