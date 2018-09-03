# coding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import cv2
import numpy as np
import pandas as pd 
from glob import glob
from tqdm import tqdm
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input

import matplotlib.pyplot as plt

model = load_model('vgg_all.h5')

im_size = (256, 256)
files = glob('../data/xuelang_round1_test_b/*.jpg')

preds = []

for file in tqdm(files):
    # print file
    im = cv2.imread(file)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    ims = []
    # for i in xrange(5):
    #     for j in xrange(5):
    #         crop_im = im[i*384:(i+1)*384,j*512:(j+1)*512,:]
    for x in xrange(0, 2560-128, 128):
        for y in xrange(0, 1920-128, 128):
            crop_im = im[y:y+256,x:x+256,:]
            ims.append(crop_im)
            flip1 =  cv2.flip(crop_im, 1)
            ims.append(flip1)
            flip2 = cv2.flip(crop_im, 0)
            ims.append(flip2)
            flip3 = cv2.flip(crop_im, -1)
            ims.append(flip3)

    ims = np.array(ims)
    ims = preprocess_input(ims)

#     x = np.expand_dims(im, axis=0)
    # pred = float(np.max(model.predict(ims)))
    pred = model.predict(ims)
    pred = zip(*([iter(np.squeeze(pred).tolist())] * 4))  
    pred = max(map(sum, pred)) / 4

    preds.append(np.clip(pred, 0.00001, 0.99999))

df = pd.DataFrame([os.path.split(f)[-1] for f in files], columns=["filename"])
df['probability'] = preds
df.to_csv("submission.csv", index=False)
