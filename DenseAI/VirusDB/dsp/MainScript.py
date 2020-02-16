# -*- coding:utf-8 -*-

import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt

from DenseAI.VirusDB.dsp.readFasta import readFasta
from DenseAI.VirusDB.dsp.numMapping import numMappingPP

def lengthCalc(Seq:list):
    max_len = 0
    min_len = 0
    mean_len = 0
    med_len = 0

    seq_lens = []
    for gene_seq in Seq:
        seq_len = len(gene_seq)
        seq_lens.append(seq_len)
    seq_lens = np.array(seq_lens)
    max_len = np.max(seq_lens)
    min_len = np.min(seq_lens)
    mean_len = int(np.round(np.mean(seq_lens)))
    med_len = int(np.round(np.median(seq_lens)))

    return max_len, min_len, mean_len, med_len

def _pad_sequences(maxlen, seq, pad_x=0):
    """
    """
    pads_x = []
    x = list(seq)
    if len(x) < maxlen:
        pads_x = x + [pad_x] * (maxlen - len(x))
    else:
        pads_x = x[0:maxlen]
    return np.array(pads_x)


def main(dataset:str):
    AcNmb, Seq, numberOfClusters, clusterNames, pointsPerCluster = readFasta(dataset)
    max_len, min_len, mean_len, med_len = lengthCalc(Seq)

    seq_fft = []
    seq_lg = []
    for gene_seq in Seq:
        num_seq = numMappingPP(gene_seq)
        num_seq = _pad_sequences(med_len, num_seq)
        # fourier transform
        num_seq_fft = fft(num_seq)
        # magnitude spectra
        num_seq_lg = np.abs(num_seq_fft)

        seq_fft.append(num_seq_fft)
        seq_lg.append(num_seq_lg)

    seq_fft = np.array(seq_fft)
    seq_lg = np.array(seq_lg)

    plt.subplot(221)
    for ii in range(1):
        plt.plot(seq_lg[ii][10:250])

    for ii in range(1):
        plt.plot(seq_lg[100+ii][10:250])
    plt.title('Original wave')
    plt.show()



    print(max_len, min_len, mean_len, med_len)

    print()

if __name__ == '__main__':

    dataset = '/home/huanghaiping/Research/Software/MLDSP-master/DataBase/Primates.mat'

    main(dataset)