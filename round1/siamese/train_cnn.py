#coding=utf-8
import os
import sys
reload(sys) 
sys.setdefaultencoding("utf-8")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

from glob import glob
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD, Adam 
from keras.utils import multi_gpu_model
from data_utils import sample_gen, gen
from model_utils import build_model, identity_loss

from config import *


data = glob('../data/xuelang_round1_train_*/*/*.jpg')

train, test = train_test_split(data, test_size=0.1, shuffle=True)
file_id_mapping_train = {f: 'normal' if u'正常' in f else 'defect' for f in train}
file_id_mapping_test = {f: 'normal' if u'正常' in f else 'defect' for f in test}

file_id_mapping_all = {f: 'normal' if u'正常' in f else 'defect' for f in data}


train_gen = sample_gen(file_id_mapping_train)
# print gen(train_gen, batch_size).next()
test_gen = sample_gen(file_id_mapping_test)

all_gen = sample_gen(file_id_mapping_all)

model = build_model()
model = multi_gpu_model(model, gpus=3)
model.compile(loss=identity_loss, optimizer=Adam(0.000001))

# checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# callbacks_list = [checkpoint, early, reduce_lr]  # early

# history = model.fit_generator(gen(train_gen, batch_size), validation_data=gen(test_gen, batch_size), epochs=60, verbose=1, workers=4, use_multiprocessing=True,
#                               callbacks=callbacks_list, steps_per_epoch=500, validation_steps=30)
                              



# model.compile(loss=identity_loss, optimizer=SGD(0.000001))
# history = model.fit_generator(gen(train_gen, batch_size), validation_data=gen(test_gen, batch_size), epochs=60, verbose=1, workers=4, use_multiprocessing=True,
#                               callbacks=callbacks_list, steps_per_epoch=500, validation_steps=30)
                              


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)

callbacks_list = [reduce_lr]  # early

history = model.fit_generator(gen(all_gen, batch_size), epochs=60, verbose=1, workers=4, use_multiprocessing=True,
                              callbacks=callbacks_list, steps_per_epoch=500)
                              



model.compile(loss=identity_loss, optimizer=SGD(0.000001))
history = model.fit_generator(gen(all_gen, batch_size), epochs=60, verbose=1, workers=4, use_multiprocessing=True,
                              callbacks=callbacks_list, steps_per_epoch=500)
                              
model.save('vgg_all.h5')