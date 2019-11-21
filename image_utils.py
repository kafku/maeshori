# coding: utf-8


import numpy as np
import pandas as pd
import math
from multiprocessing import Lock
from scipy.misc import imread, imresize
from keras.utils import np_utils, Sequence
#from sklearn.preprocessing import OneHotEncoder


class _Blacklist(object):
    def __init__(self, path):
        self.path = path
        self.blacklist = pd.DataFrame(columns=['path'])

    def __enter__(self):
        pass

    def __exit__(self, _type, value, traceback):
        if len(self.blacklist) > 0:
            with open(self.path, 'w') as f:
                self.blacklist.to_csv(f, header=False)

    def __del__(self):
        if len(self.blacklist) > 0:
            with open(self.path, 'w') as f:
                self.blacklist.to_csv(f, header=False, index=False)

    def append(self, new_path):
        self.blacklist = self.blacklist.append(
            pd.DataFrame({'path': np.atleast_1d(new_path)}))

def image_generator(img_list, image_size, num_classes, batch_size=200, shuffle=True,
                    dtype="float32", img_process=lambda x: x,
                    blacklist='./blacklist.txt'):
    """
    """
    #enc = OneHotEncoder(num_classes, dtype=np.int32, sparse=False)

    img_list = pd.DataFrame(img_list)
    bl = _Blacklist(blacklist)
    corrupted_image = []

    while True:
        if shuffle:
            img_idx = np.random.permutation(img_list.index)
        else:
            img_idx = img_list.index

        for i in range(0, len(img_list), batch_size):
            img_batch_idx = img_idx[i:min(i + batch_size, len(img_list))]

            x_train = [] # raw image
            for idx in img_batch_idx:
                try:
                    #img_data = img_to_array(load_img(img_list.loc[idx]['path'],
                    #                                 target_size=image_size))
                    img_data = imread(img_list.loc[idx]['path'], mode='RGB')
                    img_data = imresize(img_data, image_size, interp='bicubic')
                    x_train.append(img_data.astype(dtype))
                except:
                    corrupted_image.append(idx)
                    bl.append(img_list.loc[idx]['path'])
                    continue

            y_train = img_list.loc[img_batch_idx]["class"].values

            yield img_process(np.stack(x_train).astype(dtype)), np_utils.to_categorical(y_train, num_classes)
            #yield img_process(np.stack(x_train).astype(dtype)), enc.fit_transform(y_train.reshape(len(y_train), 1))

        # remove corrupted file
        if len(corrupted_image) > 0:
            img_list.drop(corrupted_image, inplace=True)
            corrupted_image = []


class ImageSequence(Sequence):
    def __init__(self, img_list, image_size, num_classes, batch_size=64,
                 dtype="float32", img_process=lambda x: x,
                 blacklist='./blacklist.txt'):
        self.img_list = pd.DataFrame(img_list)
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dtyp = dtype
        self.img_process = img_process
        self.lock = Lock()
        self.bl = _Blacklist(blacklist)
        self.corrupted_image = []

    def __len__(self):
        return math.ceil(len(self.img_list) / self.bach_size)

    def __getitem__(self, i):
        img_idx = self.img_list.index
        img_batch_idx = img_idx[i:min(i + self.batch_size, len(self.img_list))]

        x_train = []  # raw images
        for idx in img_batch_idx:
            try:
                img_data = imread(self.img_list.loc[idx]['path'], mode='RGB')
                img_data = imresize(img_data, self.image_size, interp='bicubic')
                x_train.append(img_data.astype(self.dtype))
            except Exception:
                with self.lock:
                    self.corrupted_image.append(idx)
                    self.bl.append(self.img_list.loc[idx]['path'])

                continue

        y_train = self.img_list.loc[img_batch_idx]["class"].values

        return (self.img_process(np.stack(x_train).astype(self.dtype)),
                np_utils.to_categorical(y_train, self.num_classes))

    def on_epoch_end(self):
        if self.corrupted_image:
            self.img_list.drop(self.corrupted_image, inplace=True)
            self.corrupted_image = []
