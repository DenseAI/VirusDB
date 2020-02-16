# -*- coding:utf-8 -*-

import numpy as np

# Randhawa, G., Hill, K. & Kari, L.
# ML-DSP: Machine Learning with Digital Signal Processing for ultrafast, accurate, and scalable genome classification
# at all taxonomic levels. BMC Genomics 20, 267 (2019).
# http://www.biorxiv.org/content/early/2017/07/10/161851

def numMappingAT_CG(seq: str):
    """
    Mapping ATCG to numeric, PairedNumeric representation
    :param seq:
    :return:
    """
    seq_len = len(seq)
    numSeq = np.zeros(seq_len, dtype=np.float)
    for ii in range(seq_len):
        token = seq[ii]
        if token == 'A':
            numSeq[ii] = 1
        elif token == 'C':
            numSeq[ii] = -1
        elif token == 'G':
            numSeq[ii] = -1
        elif token == 'T':
            numSeq[ii] = 1
    return numSeq


def numMappingAtomic(seq: str):
    """
    Mapping ATCG to numeric, Atomic representation
    :param seq:
    :return:
    """
    seq_len = len(seq)
    numSeq = np.zeros(seq_len, dtype=np.float)
    for ii in range(seq_len):
        token = seq[ii]
        if token == 'A':
            numSeq[ii] = 70
        elif token == 'C':
            numSeq[ii] = 58
        elif token == 'G':
            numSeq[ii] = 78
        elif token == 'T':
            numSeq[ii] = 66
    return numSeq


def numMappingCodons(seq: str):
    """
    Mapping ATCG to numeric, Codon representation
    :param seq:
    :return:
    """
    seq_len = len(seq)
    numSeq = np.zeros(seq_len, dtype=np.float)

    codons = ['TTT', 'TTC', 'TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG', 'TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC', 'TAT',
              'TAC', 'TAA', 'TAG', 'TGA', 'TGT', 'TGC', 'TGG', 'CCT', 'CCC', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG',
              'CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT',
              'AAC', 'AAA', 'AAG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA', 'GAG',
              'GGT', 'GGC', 'GGA', 'GGG']
    for ii in range(seq_len):
        token = ''
        if ii < seq_len - 2:
            token = seq[ii: ii + 3]
        elif ii == seq_len - 2:
            token = seq[ii:] + seq[0]
        else:
            token = seq[ii:] + seq[0:2]

        index = -1
        if token in codons:
            index = codons.index(token)
        numSeq[ii] = index
    return numSeq

def __cmp__(s1, s2):
    if s1 < s2:
        return -1
    elif s1 > s2:
        return 1
    else:
        return 0

def strcmp(str1, str2):
    i = 0
    while i < len(str1) and i < len(str2):
        outcome = __cmp__(str1[i], str2[i])
        if outcome:
            return outcome
        i += 1
    return __cmp__(len(str1), len(str2))

def numMappingPP(seq: str):
    """
    Mapping ATCG to numeric, Purine/Pyrimidine representation
    :param seq:
    :return:
    """
    seq_len = len(seq)
    numSeq = np.zeros(seq_len, dtype=np.float)
    for ii in range(seq_len):
        token = seq[ii]
        if strcmp(token, 'A'):
            numSeq[ii] = 1
        elif strcmp(token, 'C'):
            numSeq[ii] = -1
        elif strcmp(token, 'G'):
            numSeq[ii] = -1
        elif strcmp(token, 'T'):
            numSeq[ii] = 1
    return numSeq




if __name__ == '__main__':
    seq = 'GTTAATGTAGCTTATAATAAAGCAAAGCACTGAAAATGCTTAGATGGATTCAAAAATCCCATAAACACAA'
    numSeq = numMappingAT_CG(seq)
    print('numSeq: ', numSeq)

    numSeq = numMappingAtomic(seq)
    print('numSeq: ', numSeq)

    numSeq = numMappingCodons(seq)
    print('numSeq: ', numSeq)

    numSeq = numMappingPP(seq)
    print('numSeq: ', numSeq)