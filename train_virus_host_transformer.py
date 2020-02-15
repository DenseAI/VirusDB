# -*- coding:utf-8 -*-


import os
import math
import json
import random
import numpy as np

import keras
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

from DenseAI.VirusDB.common.callbacks import LRSchedulerPerStep, SingleModelCK
from DenseAI.VirusDB.common.utils import save_dictionary, save_word_dictionary, load_dictionaries
from DenseAI.VirusDB.model.virus_host_transformer import get_or_create, save_config
from DenseAI.VirusDB.data_loader.dataset_loader import load_data

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

max_features = 20000
maxlen = 150
num_classes = 5


def load_dictionary(config_path, encoding="utf-8"):
    '''
    Load dict
    :param config_path:
    :param encoding:
    :return:
    '''
    with open(config_path, mode="r", encoding=encoding) as file:
        str = file.read()
        config = json.loads(str)
        return config


################################################################################
# Prepare data
################################################################################
def prepare_data(train_paths:list, valid_paths:list=None, test_paths:list=None, max_seq_len:int=150, padding:bool=False):


    (x_train, y_train), (_, _), (_, _), num_words, num_labels, word_dict, label_dict = load_data(train_paths, None, None, max_seq_len=max_seq_len, padding=padding)

    max_features = num_words + 5
    max_labels = num_labels + 5

    print(x_train.shape)
    #print(x_test.shape)
    return (x_train, y_train), (_, _), (_, _), max_features, max_labels, word_dict, label_dict


EPOCH_SEC_LEN = 30  # seconds
SAMPLING_RATE = 256


if __name__ == '__main__':
    model_name = "virus_host_transformer_6gram_6step"
    config_save_path = "./data/{}_default_config.json".format(model_name)  # config path
    model_path = "./data/{}_weights.hdf5".format(model_name)
    word_dict_path = "./data/{}_word_dict.json".format(model_name)  # 目标字典路径
    label_dict_path = "./data/{}_label_dict.json".format(model_name)  # 目标字典路径

    batch_size = 32
    epochs = 150
    num_gpu = 1
    max_seq_len = 150
    initial_epoch = 0
    load = False

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    steps_per_epoch = 2000
    validation_steps = 20


    train_paths = [
        "E:\\Research\\Medical\\Data\\virus_host_db.txt",
    ]

    (x_train, y_train), (_, _),  (_, _), max_features, max_labels, word_dict, label_dict = prepare_data(train_paths, max_seq_len=max_seq_len, padding=True)
    print("x_train: ", x_train.shape)

    config = {
        'src_vocab_size': max_features,
        'tgt_vocab_size': max_labels,
        'max_seq_len': max_seq_len,
        'max_depth': 2,
        'model_dim': 128,
        'embedding_size_word': 100,
        'embedding_dropout': 0.2,
        'residual_dropout': 0.2,
        'attention_dropout': 0.1,
        'l2_reg_penalty': 0.00005,
        'confidence_penalty_weight': 0.1,
        'compression_window_size': None,
        'num_heads': 2
    }

    num_classes = max_labels

    classifer = get_or_create(config,
                              optimizer=Adam(1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                              src_dict_path=None,
                              weights_path=None,
                              num_gpu=num_gpu)

    save_config(classifer, config_save_path)
    save_word_dictionary(word_dict, word_dict_path)
    save_word_dictionary(label_dict, label_dict_path)

    classifer.model.summary()

    if load:
        classifer.parallel_model.load_weights(model_path, by_name=True);

    filepath = "./data/{}_weights.hdf5".format(model_name)

    log = TensorBoard(log_dir='./logs',
                      histogram_freq=0,
                      batch_size=batch_size,
                      write_graph=True,
                      write_grads=False)

    lr_scheduler = LRSchedulerPerStep(classifer.model_dim,
                                      warmup=2500,
                                      initial_epoch=initial_epoch,
                                      steps_per_epoch=steps_per_epoch)

    loss_name = "val_accuracy"
    mc = ModelCheckpoint(filepath, monitor=loss_name, save_best_only=True, verbose=1)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)

    print("Training")
    model_train_history = classifer.parallel_model.fit([x_train], y_train,
                                                       epochs=epochs,
                                                       batch_size=batch_size,
                                                       validation_data=([x_test], y_test),
                                                       callbacks=[lr_scheduler, mc],
                                                       initial_epoch=initial_epoch,
                                                       verbose=1)
