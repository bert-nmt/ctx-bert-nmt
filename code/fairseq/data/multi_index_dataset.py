import torch
import pdb
import itertools
# import numpy as np

from . import FairseqDataset


class MultiIndexDataset(FairseqDataset):
    """Data and indexes"""
    def __init__(self, data, indexes):
        self.data = data
        self.indexes = indexes
        # pdb.set_trace()
        assert max([max(x) for x in indexes]) < len(data)
        self._sizes = []
        for x in indexes:
            self._sizes.append([self.data.num_tokens(xx) for xx in x])

    def __getitem__(self, i):
        indexes = self.indexes[i]
        return dict(
            indexes=torch.LongTensor(indexes),
            data=[self.data[x] for x in indexes]
        )

    def __len__(self):
        return len(self.indexes)

    # def collater(self, samples):
    #     """Merge a list of samples to form a mini-batch.

    #     Args:
    #         samples (List[dict]): samples to collate

    #     Returns:
    #         dict: a mini-batch suitable for forwarding with a Model
    #     """
    #     raise NotImplementedError

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError
        # return self._sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self._sizes[index]

    @property
    def sizes(self):
        return self._sizes

    # def ordered_indices(self):
    #     """Return an ordered list of indices. Batches will be constructed based
    #     on this order."""
    #     raise NotImplementedError

    @property
    def supports_prefetch(self):
        return self.data.supports_prefetch

    def prefetch(self, indices):
        # pdb.set_trace()
        new_indices = list(itertools.chain(*[self.indexes[i] for i in indices]))
        new_indices = list(set(new_indices))
        self.data.prefetch(new_indices)

