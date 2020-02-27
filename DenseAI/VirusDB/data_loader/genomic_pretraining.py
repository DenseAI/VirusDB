
from fastai import *
from fastai.text import *
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation
import networkx as nx

from DenseAI.VirusDB.common.refseq_utils import *


class GenomicTextLMDataBunch(TextLMDataBunch):
    @classmethod
    def from_df(cls, path: PathOrStr, train_df: DataFrame, valid_df: DataFrame, test_df: Optional[DataFrame] = None,
                tokenizer: Tokenizer = None, vocab: Vocab = None, classes: Collection[str] = None,
                text_cols: IntsOrStrs = 1,
                label_cols: IntsOrStrs = 0, label_delim: str = None, chunksize: int = 10000, max_vocab: int = 60000,
                min_freq: int = 2, mark_fields: bool = False, bptt=70, collate_fn: Callable = data_collate, bs=64,
                **kwargs):
        "Create a `TextDataBunch` from DataFrames. `kwargs` are passed to the dataloader creation."
        processor = get_genomic_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                           min_freq=min_freq, mark_fields=mark_fields)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_for_lm()
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        d1 = src.databunch(**kwargs)

        datasets = cls._init_ds(d1.train_ds, d1.valid_ds, d1.test_ds)
        val_bs = bs
        datasets = [
            LanguageModelPreLoader(ds, shuffle=(i == 0), bs=(bs if i == 0 else val_bs), bptt=bptt, backwards=False)
            for i, ds in enumerate(datasets)]
        dls = [DataLoader(d, b, shuffle=False) for d, b in zip(datasets, (bs, val_bs, val_bs, val_bs)) if d is not None]

        return cls(*dls, path=path, collate_fn=collate_fn, no_check=False)

if __name__ == '__main__':

    path = Path('../../Data/Ecoil/')
    df = pd.read_csv(path / 'e_coli_lm_data.csv')

    train_df, valid_df = split_data(df, 0.9)
    tok = Tokenizer(GenomicTokenizer, n_cpus=1, pre_rules=[], post_rules=[], special_cases=['xxpad'])

    data = GenomicTextLMDataBunch.from_df(path, train_df, valid_df, bs=428, tokenizer=tok, text_cols=0, label_cols=1)

    print(data.train_ds[0])


    np.save(path / 'coli_vocab.npy', data.vocab.itos)


