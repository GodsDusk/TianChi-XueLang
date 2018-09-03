# -*- coding:utf-8 -*-
import os
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from PIL import Image
from imgaug import augmenters as iaa 


from config import *

train_seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Flipud(0.5),
            iaa.Affine(
                scale={"x": (1.0, 1.2), "y": (1.0, 1.2)},
            )
        ], random_order=True) # apply augmenters in random order



class sample_gen(object):
    def __init__(self, file_class_mapping):
        self.file_class_mapping= file_class_mapping
        self.class_to_list_files = defaultdict(list)
        
        self.list_all_files = file_class_mapping.keys()
        self.range_all_files = range(len(self.list_all_files))

        for file, class_ in file_class_mapping.items():

            self.class_to_list_files[class_].append(file)

        self.list_classes = list(set(self.file_class_mapping.values()))
        self.range_list_classes= range(len(self.list_classes))
        self.class_weight = np.array([len(self.class_to_list_files[class_]) for class_ in self.list_classes])
        self.class_weight = 1.0*self.class_weight/np.sum(self.class_weight) #new_whale:0


    def get_sample(self):

        class_idx = np.random.choice(self.range_list_classes, 1, p=self.class_weight)[0]
        examples_class_idx = np.random.choice(range(len(self.class_to_list_files[self.list_classes[class_idx]])), 2)

        positive_example_1, positive_example_2 = \
            self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[0]],\
            self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[1]]


        negative_example = None
        while negative_example is None or self.file_class_mapping[negative_example] == \
                self.file_class_mapping[positive_example_1]:
            negative_example_idx = np.random.choice(self.range_all_files, 1)[0]
            negative_example = self.list_all_files[negative_example_idx]
        return positive_example_1, negative_example, positive_example_2 

def augment(im):

    return train_seq.augment_image(im)


def read_and_resize(filepath, aug=False):
    im = cv2.imread(filepath)#(1920, 2560, 3)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, input_shape[::-1], interpolation=cv2.INTER_AREA)
    
    if aug:
        im = augment(im)
    
    return np.array(im / (np.max(im)+ 0.001), dtype="float32")


def gen(triplet_gen, batch_size):
    while True:
        list_positive_examples_1 = []
        list_negative_examples = []
        list_positive_examples_2 = []

        for i in xrange(batch_size):
            positive_example_1, negative_example, positive_example_2 = triplet_gen.get_sample()
            positive_example_1_img, negative_example_img, positive_example_2_img = read_and_resize(positive_example_1, True), \
                                                                       read_and_resize(negative_example, True), \
                                                                       read_and_resize(positive_example_2, True)


            list_positive_examples_1.append(positive_example_1_img)
            list_negative_examples.append(negative_example_img)
            list_positive_examples_2.append(positive_example_2_img)

        list_positive_examples_1 = np.array(list_positive_examples_1)
        list_negative_examples = np.array(list_negative_examples)
        list_positive_examples_2 = np.array(list_positive_examples_2)
        yield [list_positive_examples_1, list_negative_examples, list_positive_examples_2], np.ones(batch_size)



def data_generator(fpaths, batch=16):
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        img = read_and_resize(path)

        imgs.append(img)
        fnames.append(os.path.basename(path))
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)
        yield fnames, imgs
    raise StopIteration()



if __name__ == '__main__':
    import glob 
    import matplotlib.pyplot as plt
    import sys
    reload(sys) 
    sys.setdefaultencoding("utf-8")
    data = glob.glob('../data/xuelang_round1_train_*/*/*.jpg')

    train, test = train_test_split(data, test_size=0.1, shuffle=True)
    file_id_mapping_train = {f: 'normal' if u'正常' in f else 'defect' for f in train}
    file_id_mapping_test = {f: 'normal' if u'正常' in f else 'defect' for f in test}

    train_gen = sample_gen(file_id_mapping_train)
    for im, l, f in gen(train_gen, 1):
        a, b, c = im 
        print f[0], f[1], f[2]
        plt.subplot(131)
        plt.imshow(a[0])
        plt.subplot(132)
        plt.imshow(b[0])
        plt.subplot(133)
        plt.imshow(c[0])
        plt.show()