# coding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from glob import glob
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

from data_utils import SequenceData

filenames = shuffle(glob('../data/xuelang_round1_train_*/*/*.jpg'))

batch_size = 32
im_size = (256, 256)

train, test = train_test_split(filenames, test_size=0.1, shuffle=True)

train_gen = SequenceData(train, batch_size, im_size)
test_gen = SequenceData(test, batch_size, im_size)

all_gen = SequenceData(filenames, batch_size, im_size)

base_model = VGG19(weights='imagenet', include_top=False)
for layer in base_model.layers[:-2]:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)



model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]  # early


# model.fit_generator(train_gen, steps_per_epoch=len(filenames)/batch_size,epochs=100,validation_data=test_gen, workers=4, use_multiprocessing=True, shuffle=True, class_weight = 'auto', callbacks = callbacks_list)
model.fit_generator(all_gen, steps_per_epoch=len(filenames)/batch_size,epochs=100, workers=4, use_multiprocessing=True, shuffle=True, class_weight = 'auto')
model.save('vgg_all.h5')