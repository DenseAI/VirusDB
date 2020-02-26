

# from fastai import *
# from fastai.text import *

from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation
import networkx as nx

from DenseAI.VirusDB.common.refseq_utils import *


# Extract promoter regions
# All sequences saved as sense strands
def extract_promoter(feature):
    quals = list(feature.qualifiers.keys())

    if 'gene' in quals:
        gene = feature.qualifiers['gene']
    else:
        gene = None

    if 'locus_tag' in quals:
        locus = feature.qualifiers['locus_tag']
    else:
        locus = None

    cds_loc = str(feature.location)
    cds_start = feature.location.start.position
    cds_end = feature.location.end.position
    p_start = cds_start - 100
    p_end = cds_end + 100

    if feature.strand == -1:
        orient = 'reverse'
        promoter = GB[cds_end - 50:p_end].reverse_complement().seq.__str__()
        promoter_loc = f"[{cds_end - 50}:{p_end}]" + f"{cds_loc[-3:]}"

    else:
        orient = 'forward'
        promoter = GB[p_start:cds_start + 50].seq.__str__()
        promoter_loc = f"[{p_start}:{cds_start + 50}]" + f"{cds_loc[-3:]}"

    promoter_data = [gene, locus, cds_loc, promoter_loc, orient, promoter, 1]

    return promoter_data


# Extract negative examples
def extract_gene(feature):
    quals = list(feature.qualifiers.keys())

    if 'gene' in quals:
        gene = feature.qualifiers['gene']
    else:
        gene = None

    if 'locus_tag' in quals:
        locus = feature.qualifiers['locus_tag']
    else:
        locus = None

    cds_loc = str(feature.location)
    cds_start = feature.location.start.position
    cds_end = feature.location.end.position

    gene_len = (cds_end - 50) - (cds_start + 50)

    if gene_len > 150:
        rand_start = np.random.randint(cds_start + 50, cds_end - 200)
        rand_gene = GB[rand_start:rand_start + 150]
        rand_gene_loc = f"[{rand_start}:{rand_start + 150}]" + f"{cds_loc[-3:]}"

        if feature.strand == -1:
            orient = 'reverse'
            rand_gene = rand_gene.reverse_complement().seq.__str__()

        else:
            orient = 'forward'
            rand_gene = rand_gene.seq.__str__()

        gene_data = [gene, locus, cds_loc, rand_gene_loc, orient, rand_gene, 0]
        return gene_data

    else:
        return False


if __name__ == '__main__':
    path = Path('../../Data/Ecoil/')

    fname = 'GCF_000005845.2_ASM584v2_genomic.fna'
    # data = process_fasta(path / fname, 2000, 900)
    #
    # print("data: ", data)
    #
    # val_pct = 0.15
    # cut = int(len(data) * val_pct) + 1
    #
    # train_df = pd.DataFrame(data[:cut], columns=['Sequence'])
    # valid_df = pd.DataFrame(data[cut:], columns=['Sequence'])
    # train_df['is_train'] = 1
    # valid_df['is_train'] = 0
    #
    # data_df = pd.concat([train_df, valid_df])
    # data_df.to_csv(path / 'e_coli_lm_data.csv', index=False)

    fname = 'GCF_000005845.2_ASM584v2_genomic.gbff'

    GB = next(SeqIO.parse(path / fname, "genbank"))

    proms = []
    for feature in GB.features:
        if feature.type == 'CDS':
            proms.append(extract_promoter(feature))

    prom_df = pd.DataFrame(proms, columns=['Gene', 'Locus', 'Location', 'Sample Location', 'Orientation', 'Sequence',
                                           'Promoter'])

    genes = []
    for feature in GB.features:
        if feature.type == 'CDS':
            gene = extract_gene(feature)
            if gene:
                genes.append(gene)

    gene_df = pd.DataFrame(genes, columns=['Gene', 'Locus', 'Location', 'Sample Location', 'Orientation', 'Sequence',
                                           'Promoter'])

    promoter_data = pd.concat([prom_df, gene_df])

    promoter_data.to_csv(path / 'e_coli_promoters.csv', index=False)




