# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging
import json


def _parse_db_text(file_name: str, word_index_from=5, label_index_from=3, lower=True, sent_delimiter='\t'):
    """
    Parse virushostdb.tsv
    """
    virus_entities = []
    refseq_set = set()
    if os.path.exists(file_name) is False:
        logging.error("File is not exists: {}".format(file_name))
        return virus_entities, refseq_set

    try:
        file = open(file_name, 'r', encoding="utf-8")
        index = 0
        for line in file:

            if index == 0:
                index += 1
                continue

            # Replace '\n'
            if len(line) > 0:
                line = line[:-1]

            # virus tax id,
            # virus name,
            # virus lineage,
            # refseq id,
            # KEGG GENOME,
            # KEGG DISEASE,
            # DISEASE
            # host tax id
            # host name
            # host lineage
            # pmid
            # evidence
            # sample type
            # source organism
            # 存在 一个 virus tax id 对多个 refseq id 和 对个 host tax id的场景
            tokens = line.split(sent_delimiter)
            if len(tokens) < 14:
                continue

            virus_tax_id = tokens[0]

            refseq_ids = tokens[3]
            refseq_tokens = refseq_ids.split(',')
            if refseq_tokens is None or len(refseq_tokens) == 0:
                continue

            host_tax_id = tokens[7]

            for refseq in refseq_tokens:
                refseq = refseq.strip()
                refseq = refseq.lower()
                virus_entity = {}
                virus_entity['virus_tax_id'] = virus_tax_id
                virus_entity['virus name'] = tokens[1]
                virus_entity['virus lineage'] = tokens[2]
                virus_entity['refseq_id'] = refseq
                virus_entity['host_tax_id'] = host_tax_id
                virus_entity['host name'] = tokens[8]
                virus_entity['host lineage'] = tokens[9]
                virus_entity['DISEASE'] = tokens[6]
                virus_entities.append(virus_entity)

                refseq_set.add(refseq.lower())
                #print(virus_entity)
                #print()
            index += 1
    except Exception as e:
        logging.error(e)

    return virus_entities, refseq_set



def _parse_genomic_text(file_name: str, refseq_set:set=None, word_index_from:int=5, label_index_from:int=3, lower:bool=True, sent_delimiter:str='\t'):
    """
    Parse virushostdb.tsv
    """
    refseq_entities = {}
    if os.path.exists(file_name) is False:
        logging.error("File is not exists: {}".format(file_name))
        return refseq_entities

    try:
        file = open(file_name, 'r', encoding="utf-8")
        index = 0

        refseq_id = ''
        refseq = ''
        for line in file:
            # Replace '\n'
            if len(line) > 0:
                line = line[:-1]

            if line.startswith('>'):
                # Replace '>' and '||'
                line = line[1:-2]

                if len(refseq_id) > 0 and len(refseq) > 0:
                    refseq_id = refseq_id.lower()

                    # Check refseq_id
                    if refseq_set is not None and refseq_id in refseq_set:
                        refseq_entities[refseq_id] = refseq
                        if len(refseq_entities) == 10:
                            print(refseq)
                    else:
                        print(refseq_id)

                refseq_id = ''
                refseq = ''
                # Sequence_accession virus name
                # Hostname
                # Virus lineage
                # Host lineage
                # Sample type
                # Taxonomic identifier
                tokens = line.split('|')
                if len(tokens) < 1:
                    continue

                # refseq_id
                for ii in range(len(tokens[0])):
                    if tokens[0][ii] == ' ':
                        break
                    refseq_id += tokens[0][ii]

                #print(refseq_id)
            else:
                refseq += line
            index += 1

        if len(refseq_id) > 0 and len(refseq) > 0:
            refseq_id = refseq_id.lower()
            # Check refseq_id
            if refseq_set is not None and refseq_id in refseq_set:
                refseq_entities[refseq_id] = refseq

    except Exception as e:
        logging.error(e)

    print(index)
    return refseq_entities


if __name__ == '__main__':
    train_paths = [
        "F:\\Research\\Data\\medical\\train.txt",
        # "E:\\Research\Corpus\\pku_training_bies",
        # "E:\\Research\Corpus\\weibo_train_bies",
        # "E:\\Research\Corpus\\people_2014_train_bies",
        # "E:\\Research\Corpus\\people_1998_train_bies",
    ]

    virushostdb_file = '/home/huanghaiping/Research/Medical/VirusHostDb/virushostdb.tsv'
    virus_entities, refseq_set = _parse_db_text(virushostdb_file)

    virus_genomic_file = '/home/huanghaiping/Research/Medical/VirusHostDb/virushostdb.formatted.genomic.fna'
    refset_entities = _parse_genomic_text(virus_genomic_file, refseq_set=refseq_set)

    print(len(refset_entities))

    seq_lens = []

    output_file = '/home/huanghaiping/Research/Medical/Data/virus_host_db.txt'
    with open(output_file, "w") as fh:
        for virus_entity in virus_entities:
            refseq_id = virus_entity['refseq_id']
            refseq_id = refseq_id.lower()
            if refseq_id not in refset_entities.keys():
                print(virus_entity)

            virus_entity['refseq'] = refset_entities[refseq_id]

            seq_len = len(refset_entities[refseq_id])
            seq_lens.append(seq_len)
            if seq_len > 40 * 1024:
                print(seq_len)
            fh.write(json.dumps(virus_entity) + '\n')



    seq_lens = np.array(seq_lens)
    print("Max len: ", max(seq_lens))
    print("Min len: ", min(seq_lens))
    print("Mean len: ", np.mean(seq_lens))

