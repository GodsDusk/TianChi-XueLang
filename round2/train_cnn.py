#coding=utf-8
import os
import sys
reload(sys) 
sys.setdefaultencoding("utf-8")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import pickle
import copy
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean 
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD, Adam 
from keras.utils import multi_gpu_model
from data_utils import sample_gen, gen, data_generator,hard_sample_gen
from model_utils import build_model, identity_loss

from config import *


data = pd.read_csv('../train.csv')

train, test = train_test_split(data, test_size=0.1, shuffle=True)

file_id_mapping_train = {k: v for k, v in zip(train.images.values, train.classes.values)}
file_id_mapping_test = {k: v for k, v in zip(test.images.values, test.classes.values)}



file_id_mapping_all =  {k: v for k, v in zip(data.images.values, data.classes.values)}





all_gen = sample_gen(file_id_mapping_all)

model, base_model = build_model()
model = multi_gpu_model(model, gpus=4)
model.compile(loss=identity_loss, optimizer=Adam(0.000001))





def compute_distance(file_vertors, file, files, multiple, sample_num, reverse):

    distances = [(f, euclidean(file_vertors[file],file_vertors[f])) for f in files]
    sample_num = max(int(len(distances)*0.8**multiple), 2)
    distances = sorted(distances, key=lambda distances:distances[1], reverse=reverse)[:sample_num]
    return [d[0] for d in distances]

def gen_distance(base_model, file_id_mapping, multiple):

    lable2file = defaultdict(list)
    for file,label in file_id_mapping.items():
        lable2file[label].append(file) 
    files = file_id_mapping.keys()
    vectors = []
    for fnames, imgs in tqdm(data_generator(files, batch=batch_size), total=len(files)/batch_size, desc='gen_vectors'):

        predicts = base_model.predict(imgs)
        vectors += predicts.tolist()

    file_vertors = dict(zip(files,vectors))
    file_distance = {}
    for label, files in tqdm(lable2file.items(), desc='computer distances'):
        for file in files:
            # print file in file_vertors
            file_distance[file] = {}            
            file_distance[file]['same'] = compute_distance(file_vertors, file, files, multiple, sample_num=100, reverse=True)

            temp_lf = copy.deepcopy(lable2file)
            temp_lf.pop(label)
            other_file = sum(temp_lf.values(), [])
            file_distance[file]['different'] = compute_distance(file_vertors, file, other_file, multiple, sample_num=100, reverse=False)

    return file_distance

def train_val(model, base_model):

    train_gen = sample_gen(file_id_mapping_train)
    # print gen(train_gen, batch_size).next()
    test_gen = sample_gen(file_id_mapping_test) 

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    callbacks_list = [checkpoint, early, reduce_lr]  # early


    history = model.fit_generator(gen(train_gen, batch_size), validation_data=gen(test_gen, batch_size), epochs=60, verbose=1, workers=4, use_multiprocessing=True,
                                  callbacks=callbacks_list, steps_per_epoch=500, validation_steps=30)
                                  

    # model.compile(loss=identity_loss, optimizer=SGD(0.000001))
    # history = model.fit_generator(gen(train_gen, batch_size), validation_data=gen(test_gen, batch_size), epochs=60, verbose=1, workers=4, use_multiprocessing=True,
    #                               callbacks=callbacks_list, steps_per_epoch=500, validation_steps=30)
                                  

    # return
    file_name = file_path
    for i in xrange(1,10):
        train_file_distance = gen_distance(base_model, file_id_mapping_train, i)
        test_file_distance = gen_distance(base_model, file_id_mapping_test, i)
        train_gen = hard_sample_gen(train_file_distance)
        test_gen = hard_sample_gen(test_file_distance)

        model, base_model = build_model()
        model = multi_gpu_model(model, gpus=4)
        model.compile(loss=identity_loss, optimizer=Adam(0.000001))
        model.load_weights(file_name)
        file_name = 'hard_{}.h5'.format(i)

        checkpoint = ModelCheckpoint(file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=15)
        history = model.fit_generator(gen(train_gen, batch_size), validation_data=gen(test_gen, batch_size), epochs=60, verbose=1, workers=4, use_multiprocessing=True,
                                     steps_per_epoch=500, validation_steps=30, callbacks=[checkpoint, early])


def train_all():

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)

    callbacks_list = [reduce_lr]  # early

    history = model.fit_generator(gen(all_gen, batch_size), epochs=60, verbose=1, workers=4, use_multiprocessing=True,
                                  callbacks=callbacks_list, steps_per_epoch=500)
                                  



    model.compile(loss=identity_loss, optimizer=SGD(0.000001))
    history = model.fit_generator(gen(all_gen, batch_size), epochs=60, verbose=1, workers=4, use_multiprocessing=True,
                                  callbacks=callbacks_list, steps_per_epoch=500)
                                  
    model.save('vgg_all.h5')


train_val(model, base_model)