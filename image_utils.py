# coding: utf-8


import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import load_img
#from sklearn.preprocessing import OneHotEncoder

def image_generator(img_list, image_size, num_classes, batch_size=200,
                        dtype="float32", img_process=lambda x: x):
    """
    """
    #enc = OneHotEncoder(num_classes, dtype=np.int32, sparse=False)

    while True:
        img_idx = np.random.permutation(img_list.index)

        for i in range(0, len(img_list), batch_size):
            img_batch_idx = img_idx[i:min(i + batch_size, len(img_list) - 1)]

            x_train = [] # raw image
            for img_path in img_list.loc[img_batch_idx]["path"]:
                img_data = load_img(img_path, target_size=image_size)
                x_train.append(img_data.astype(dtype))

            y_train = img_list.loc[img_batch_idx]["class"].values

            yield img_process(np.stack(x_train).astype(dtype)), np_utils.to_categorical(y_train, num_classes)
            #yield img_process(np.stack(x_train).astype(dtype)), enc.fit_transform(y_train.reshape(len(y_train), 1))
