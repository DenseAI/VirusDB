# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging
import json


SEP = "[SEP]"
CLS = "[CLS]"
MASK = "[MASK]"
UNK = "[UNK]"


def _parse_text(file_name: str, word_dict: dict, label_dict: dict, word_index_from: int = 5, label_index_from: int = 3,
                n_gram:int=6, m_step:int=6, label_name:str='host_tax_id', lower: bool = True):
    """
    Read corpus 读取语料
    :param file_name:文件名称
    :param word_index_from: 词语开始编号
    :param label_index_from: 标签开始编号
    :param lower: 转化为小写
    :param sent_delimiter: 词语分隔符
    :param padding: 是否填充
    :return:
    """

    words = []
    labels = []

    if os.path.exists(file_name) is False:
        logging.error("File is not exists: {}".format(file_name))
        return words, labels

    try:
        file = open(file_name, 'r', encoding="utf-8")
        index = 0
        for line in file:
            virus_entity = json.loads(line)
            if virus_entity is None:
                continue

            word = []
            label = []
            CLS = '[CLS]'

            genomic_seq = virus_entity['refseq']
            if len(genomic_seq) == 0:
                continue

            # Words
            if CLS not in word_dict.keys():
                word_dict[CLS] = len(word_dict) + word_index_from
            word.append(word_dict[CLS])

            #for
            for ii in range(0, len(genomic_seq), m_step):
                if ii + n_gram <=  len(genomic_seq):
                    char = ''
                    for jj in range(ii, ii + n_gram):
                        char += genomic_seq[jj]
                    if char not in word_dict.keys():
                        word_dict[char] = len(word_dict) + word_index_from
                    word.append(word_dict[char])


            # Tags
            tag = virus_entity[label_name]
            if tag not in label_dict.keys():
                label_dict[tag] = len(label_dict) + label_index_from
            label.append(label_dict[tag])

            if len(word) > 0 and len(label) > 0:
                words.append(np.array(word))
                labels.extend(np.array(label))
                # ner_labels.append(ner_tags)

            index += 1
            if index > 0 and index % 100 == 0:
                print(index)

    except Exception as e:
        logging.error(e)

    # print("words: ", words)
    # print("labels: ", labels)
    # print("ner_labels: ", len(ner_labels), ner_labels)
    return words, labels


def load_data(train_paths: list,
              valid_paths: list,
              test_paths: list,
              num_words=None,
              max_seq_len=25,
              word_index_from=5,
              label_index_from=3,
              lower=True,
              sent_delimiter='\t',
              padding=False,
              word_dict: dict = None,
              label_dict: dict = None,
              add_to_dict=True,
              **kwargs):
    """
    Load dataset 读取数据集
    """

    if word_dict is None:
        word_dict = {}
        word_dict[SEP] = 1
        word_dict[CLS] = 2
        word_dict[MASK] = 3
        word_dict[UNK] = 4

    if label_dict is None:
        label_dict = {}
        label_dict[CLS] = 1
        label_dict[UNK] = 2

    # Load Train set 读取训练语料
    x_train = []
    y_train = []
    count = 0
    if train_paths != None and len(train_paths):
        for file_name in train_paths:
            words, labels = _parse_text(file_name, word_dict, label_dict, word_index_from=word_index_from, label_index_from=label_index_from, n_gram=5, m_step=3)
            x_train.extend(words)
            y_train.extend(labels)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)

    x_valid = []
    y_valid = []
    count = 0
    if valid_paths != None and len(valid_paths) >= 0:
        for file_name in valid_paths:
            words, labels = _parse_text(file_name, word_dict, label_dict, word_index_from=word_index_from, label_index_from=label_index_from)
            x_valid.extend(words)
            y_valid.extend(labels)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    print("x_valid: ", x_valid.shape)
    print("y_valid: ", y_valid.shape)


    x_test = []
    y_test = []
    ner_labels_test = []
    count = 0
    if test_paths != None and len(test_paths) >= 0:
        for file_name in test_paths:
            words, labels = _parse_text(file_name, word_dict, label_dict, word_index_from=word_index_from, label_index_from=label_index_from)
            x_test.extend(words)
            y_test.extend(labels)
            # ner_labels_test.extend(ner_labels)

    print("Test Counter: ", count)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)

    print("word_dict: ", len(word_dict))
    print("label_dict: ", len(label_dict))

    if max_seq_len > 0 and padding:
        if len(x_train) > 0:
            x_train = _remove_long_seq(max_seq_len, x_train)
            if not x_train:
                raise ValueError('After filtering for sequences shorter than maxlen=' +
                                 str(max_seq_len) + ', no sequence was kept. '
                                               'Increase maxlen.')
            x_train = _pad_sequences(max_seq_len, x_train)
            x_train = np.array(x_train)

        if len(x_valid) > 0:
            x_valid = _remove_long_seq(max_seq_len, x_valid)
            if not x_valid:
                raise ValueError('After filtering for sequences shorter than maxlen=' +
                                 str(max_seq_len) + ', no sequence was kept. '
                                               'Increase maxlen.')
            x_valid = _pad_sequences(max_seq_len, x_valid)
            x_valid = np.array(x_valid)
            y_valid = np.array(y_valid)

        if len(x_test) > 0:
            x_test = _remove_long_seq(max_seq_len, x_test)
            if not x_test:
                raise ValueError('After filtering for sequences shorter than maxlen=' +
                                 str(max_seq_len) + ', no sequence was kept. '
                                               'Increase maxlen.')
            x_test = _pad_sequences(max_seq_len, x_test)
            x_test = np.array(x_test)
            y_test = np.array(y_test)

    if not num_words:
        num_words = len(word_dict)
        num_labels = len(label_dict)

    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_valid: ", x_valid.shape)
    print("y_valid: ", y_valid.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), num_words, num_labels, word_dict, label_dict


def _remove_long_seq(maxlen, seq):
    """Removes sequences that exceed the maximum length.

    # Arguments
        maxlen: Int, maximum length of the output sequences.
        seq: List of lists, where each sublist is a sequence.
        label: List where each element is an integer.

    # Returns
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label = [], []
    count = 0
    for x in seq:
        if len(x) < maxlen:
            new_seq.append(x)
        else:
            new_seq.append(x[0:maxlen])
            count += 1
    print("Remove: ", count)
    return new_seq  # , new_value


def _pad_sequences(maxlen, seq, pad_x=0, pad_y=0, pad_v=0):
    """Removes sequences that exceed the maximum length.

    # Arguments
        maxlen: Int, maximum length of the output sequences.
        seq: List of lists, where each sublist is a sequence.
        label: List where each element is an integer.

    # Returns
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label, new_ner = [], [], []
    for x in seq:
        x = list(x)
        if len(x) < maxlen:
            pads_x = x + [pad_x] * (maxlen - len(x))
            new_seq.append(pads_x)
        else:
            new_seq.append(x[0:maxlen])

    return new_seq



if __name__ == '__main__':
    train_paths = [
        "E:\\Research\\Medical\\Data\\virus_host_db.txt",
    ]

    (x_train, y_train), (_, _), (_, _), num_words, num_labels, word_dict, label_dict = load_data(train_paths, None, None, max_seq_len=150, padding=True)

    word_index_dict = {}
    for d, v in word_dict.items():
        word_index_dict[v] = d

    label_index_dict = {}
    for d, v in label_dict.items():
        label_index_dict[v] = d

    print("label_index_dict: ", label_index_dict)
    x_sent = ""
    y_sent = ""

    x_raw = []
    y_raw = []

    example_index = 1
    output_file = 'E:\\Research\\Medical\\Data\\virus_host_db_fasttext.txt'
    with open(output_file, "w") as fh:
        for ii in range(len(x_train)):
            x_data = x_train[ii]
            x_sent += '__label__' + label_index_dict.get(y_train[ii], '[UNK]')
            for jj in range(len(x_data)):
                x_sent += ' ' + word_index_dict.get(x_data[jj], '[UNK]')
            fh.write(x_sent + '\n')



