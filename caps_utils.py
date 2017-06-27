# coding: utf-8

import os
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
from skimage import io
from scipy.misc import imresize
from sklearn.utils import shuffle
from skimage.color import gray2rgb
from nltk.tokenize import word_tokenize
from pycocotools.coco import COCO
from keras.utils.np_utils import to_categorical
from .rnn_utils import rnn_formatter

class CocoGenerator(object):
    """
    """

    def __init__(self,
                 coco_data_path,
                 coco_data_type,
                 img_dir=None,
                 word_dict=None,
                 vocab_size=None,
                 word_dict_creator=None,
                 tokenizer=word_tokenize,
                 caps_process=lambda x: x,
                 raw_img=False,
                 on_memory=False,
                 img_size=None,
                 img_process=lambda x: x,
                 feature_extractor=None,
                 cache=None):
        """
        Args:
            coco_data_path: Path to COCO dataset
            coco_data_type: e.g. train2014, val2014
            img_dir: Path to the image directory.
                     If missing, coco_data_path/images/coco_data_type
            word_dict: precomputed wor dictionary which takes str as keys and
                       integer (index) as values.
            word_dict_creator: Function that takes text then returns word dictionary
                               and vocaburary size
           tokenizer: Tokenizer applied to the captions
           caps_process: Preprocessing function applied to the captions
           on_memory: Load all images or image features on memory in advance (untested)
        """
        # image data dir
        if img_dir is None:
            self.img_dir = "%s/images/%s/"%(coco_data_path, coco_data_type)

        # coco instances
        coco = COCO('%s/annotations/instances_%s.json'%(coco_data_path, coco_data_type))

        # coco captions
        coco_caps = COCO('%s/annotations/captions_%s.json'%(coco_data_path, coco_data_type))

        # preprocessor
        self.feature_extractor = feature_extractor
        self.img_process = img_process

        # word dictionary
        if word_dict is None:
            all_text = " ".join([caps_process(val["caption"]) for val in coco_caps.anns.values()])
            self.word_dict, self.vocab_size = word_dict_creator(all_text)
            del all_text
        else:
            self.word_dict = word_dict
            self.vocab_size = vocab_size

        # load image path
        self.img_path = dict(
            [(key, value["file_name"]) for key, value in coco.imgs.items()])

        # load caption table
        self.caption_table = pd.DataFrame([i for i in coco_caps.anns.values()])
        self.caption_table["caption"] = self.caption_table["caption"].apply(
            lambda x: [self.word_dict[token] for token in tokenizer(caps_process(x))])
        self.num_captions = self.caption_table.shape[0]

        # load all images in advance
        self.img_size = img_size
        self.on_memory = on_memory
        self.raw_img = raw_img
        if on_memory:
            if cache is not None and os.path.exists(cache):
                print('Loading %s'%cache)
                with open(cache, 'rb') as cache_file:
                    self.img_dict = pickle.load(cache_file)
            else:
                self.img_dict = dict()
                for image_id in tqdm(self.caption_table["image_id"].unique()):
                    img_data = self._load_image(image_id)
                    if raw_img:
                        self.img_dict[image_id] = img_data
                    else:
                        self.img_dict[image_id] = self.feature_extractor(img_data)
                if cache is not None:
                    print('Saving %s'%cache)
                    with open(cache, 'wb') as cache_file:
                        pickle.dump(self.img_dict, cache_file)


        if on_memory:
            self._get_img_feature = lambda image_id: self.img_dict[image_id]
        else:
            if raw_img:
                self._get_img_feature = self._load_image
            else:
                self._get_img_feature = lambda image_id: self.feature_extractor(self._load_image(image_id))

    def _load_image(self, image_id):
        """
        Args:
            image_id: COCO image ID

        Return:
            preprocessed image (numpy.array)
        """

        img_path = os.path.join(self.img_dir, self.img_path[image_id])
        img_data = imresize(io.imread(img_path), self.img_size, interp='bicubic') # channel last
        if len(img_data.shape) == 2 or img_data.shape[2] == 1: # if gray scale
            img_data = gray2rgb(img_data)
        return np.expand_dims(self.img_process(img_data), axis=0) # (bach_size=1, image_size)


    def generator(self,
                  img_key="img_input", lang_key="lang_input",
                  start_signal="<BOS>", end_signal="<EOS>",
                  img_size=None, feature_extractor=None,
                  format_split=True,
                  onehot_y=True,
                  **kwargs):
        """
        Args:
            img_key: A key values of the image feature
            lang_key: A key values of the caption data
            start_signal:
            end_signal:
            img_size: Size of images passed feature_extractor or that of oupututs.
                      This option will be ignored when on_memory is True.
            feature_extractor: Function that extracts image features.
                               This funciton takes 3-dim array then returns 1D array
                               This option will be ingnored when on_memory is True.
            kwags: Args passed to rnn_formatter
        """
        if not self.on_memory and not img_size is None:
            self.img_size = img_size

        if not self.on_memory and not feature_extractor is None:
            self.feature_extractor = feature_extractor

        # replace signals
        if not start_signal is None:
            start_signal = self.word_dict[start_signal]
        if not end_signal is None:
            end_signal = self.word_dict[end_signal]

        while True:
            for caption_info in shuffle(self.caption_table).iterrows():
                # process caption sequence
                if format_split:
                    X_lang, y_lang = rnn_formatter(caption_info[1]['caption'],
                                                   start_signal=start_signal,
                                                   end_signal=end_signal,
                                                   **kwargs)
                else:
                    x_caption = list(caption_info[1]['caption'])
                    y_caption = list(caption_info[1]['caption'])
                    if not start_signal is None:
                        x_caption.insert(0, start_signal)
                        y_caption.insert(0, start_signal)
                    if not end_signal is None:
                        y_caption.append(end_signal)

                    # padding (post)
                    if 'maxlen' in kwargs:
                        x_caption += [0] * (kwargs['maxlen'] - len(x_caption))
                        y_caption += [0] * (kwargs['maxlen'] + 1 - len(y_caption))

                    X_lang = np.atleast_2d(x_caption)
                    y_lang = np.atleast_2d(y_caption)

                y_lang = np.maximum(y_lang - 1, 0)
                if onehot_y:
                    y_lang = to_categorical(y_lang, num_classes=self.vocab_size)
                elif format_split:
                    y_lang = np.atleast_2d(y_lang).T

                # load image feature
                image_feature = self._get_img_feature(caption_info[1]["image_id"])
                new_shape = (X_lang.shape[0],) + (1, 1, 1)[0:len(image_feature.shape)]
                image_feature = np.tile(image_feature, new_shape)

                yield {img_key: image_feature, lang_key: X_lang}, y_lang
