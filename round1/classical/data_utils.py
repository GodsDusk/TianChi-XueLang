#coding=utf-8
import os
import sys
reload(sys) 
sys.setdefaultencoding("utf-8")
import cv2
import numpy as np
import xml.dom.minidom as minidom
from random import sample,choice
from keras.utils import Sequence
from keras.applications.vgg19 import preprocess_input
from imgaug import augmenters as iaa


class SequenceData(Sequence):
    def __init__(self, filenames, batch_size, im_size, view=False):

        self.view = view
        self.filenames = filenames
        self.labels = [0 if u'正常' in l else 1 for l in filenames]

        self.batch_size = batch_size
        self.im_size = im_size
        self.normal_seq = iaa.Sequential([
                        iaa.Fliplr(0.5), # horizontal flips
                        iaa.Flipud(0.5),
                        iaa.Affine(
                            scale={"x": (1.0, 1.2), "y": (1.0, 1.2)}
                        )
                    ], random_order=True) # apply augmenters in random order


    def __len__(self):
        return len(self.filenames)/self.batch_size

    def __getitem__(self, idx):

        batch_x = self.filenames[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
        if self.view:
            return np.array([self.read_im(x) for x in batch_x]), np.array(batch_y), batch_x
        else:
            return np.array([self.read_im(x) for x in batch_x]), np.array(batch_y) 

    def read_im(self, f):
        im = cv2.imread(f)#(1920, 2560, 3)
        h, w, _ = im.shape
        rw ,rh = self.im_size#(512, 384)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if u'正常' in f:
            x = np.random.randint(0, w-rw)
            y = np.random.randint(0, h-rh)
            im = im[y:y+rh,x:x+rw,:]
            
        else: 

            bboxes = self.read_xml(f)
            bbox = bboxes[0] if len(bboxes) == 1 else np.random.choice(bboxes)
            xmin, ymin, xmax, ymax = bbox
            if self.view:
                cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)
            if xmin+rw/2 < xmax-rw/2:
                x = np.random.randint(xmin+rw/2, xmax-rw/2)
            else:
                x = (xmin + xmax) / 2
            if ymin+rh/2 < ymax-rh/2:
                y = np.random.randint(ymin+rh/2, ymax-rh/2)
            else:
                y = (ymin + ymax) / 2
            x = max(0, x-rw/2)
            y = max(0, y-rh/2)
            im = im[y:y+rh,x:x+rw,:]

        im = self.normal_seq.augment_image(im)
        # print im.shape, u'正常' in f, x, y
        im = cv2.resize(im, self.im_size, interpolation=cv2.INTER_AREA)
        if not self.view:
            im = preprocess_input(im)
        
        return im

    def read_xml(self, f):

        func = lambda x:int(x[0].childNodes[0].data)

        DOMTree = minidom.parse(f.replace('jpg', 'xml'))
        annotation = DOMTree.documentElement
        objectlist = annotation.getElementsByTagName('object')
        bndbox = objectlist[0].getElementsByTagName('bndbox')
        bboxes = []
        for i in xrange(len(bndbox)):
            xmin = func(bndbox[i].getElementsByTagName('xmin')) 
            ymin = func(bndbox[i].getElementsByTagName('ymin')) 
            xmax = func(bndbox[i].getElementsByTagName('xmax')) 
            ymax = func(bndbox[i].getElementsByTagName('ymax')) 
            bboxes.append([xmin, ymin, xmax, ymax])
        return bboxes

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt 
    from glob import glob
    from collections import defaultdict
    from sklearn.utils import shuffle
    filenames = shuffle(glob('../data/xuelang_round1*/*/*.jpg'))
    print len(filenames)
    classes = defaultdict(list)
    for file in filenames:
        class_ = file.split('/')[2]
        classes[class_].append(file) 
    for i, v in classes.items():
        print i, len(v)
    gen = SequenceData(filenames, 9, (256, 256), view=True)
    for im, l, f in gen:

        print im.shape,l.shape
        for i in xrange(9):
            print f[i]
            plt.subplot(3,3,i+1)
            plt.imshow(im[i])
        plt.show()