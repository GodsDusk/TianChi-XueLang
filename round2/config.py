#coding=utf-8
import os
import sys
reload(sys) 
sys.setdefaultencoding("utf-8")

defect_dict={'正常':'norm', '扎洞':'defect_1', '毛斑':'defect_2', '擦洞':'defect_3',
             '毛洞':'defect_4', '织稀':'defect_5', '吊经':'defect_6', '缺经':'defect_7',
             '跳花':'defect_8', '油渍':'defect_9', '污渍':'defect_9'}

defect_list = set(defect_dict.values() + ['defect_10'])


input_shape = (768, 1024)
# input_shape = (600, 800)
batch_size = 4
model_name = "triplet_model_"
file_path = model_name + "weights_best_inception.hdf5"
base_path = "../input/train_aug/"
