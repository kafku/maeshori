# coding: utf-8


import numpy as np
import pandas as pd
from scipy.misc import imread, imresize
from keras.utils import np_utils
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
