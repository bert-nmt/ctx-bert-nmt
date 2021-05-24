### Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
### Written by Lesly Miculicich <lmiculicich@idiap.ch>

import torch
import numpy as np
import pdb
import itertools

from . import data_utils
from .language_pair_document_flexible_ctx import LanguagePairDocumentFlexibleCtxDataset
from .iterators import EpochBatchIterator, ShardedIterator, CountingIterator


class DocumentEpochBatchIterator(EpochBatchIterator):

    def __init__(
        self, dataset, collate_fn, batch_sampler_for_indices,
        seed=1, num_shards=1, shard_id=0,
        num_workers=0, epoch=0,
    ):
        assert isinstance(dataset, LanguagePairDocumentFlexibleCtxDataset)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers

        self.batch_sampler_for_indices = batch_sampler_for_indices

        self.epoch = epoch
        self._cur_epoch_itr = None
        self._next_epoch_itr = None
        self._supports_prefetch = getattr(dataset, 'supports_prefetch', False)

        # pdb.set_trace()
        self.doc_start = np.array(self.dataset.doc_index, dtype=np.int)
        self.doc_end = np.array(list(self.doc_start[1:]) + [len(self.dataset), ], dtype=np.int)
        self.frozen_batches = None

    def get_batches(self, shuffle=False, seed=None):
        # pdb.set_trace()
        doc_idx = list(range(len(self.doc_start)))

        if shuffle:
            with data_utils.numpy_seed(seed):
                np.random.shuffle(doc_idx)

        indices = itertools.chain(*[range(self.doc_start[i], self.doc_end[i]) for i in doc_idx])
        return list(self.batch_sampler_for_indices(indices))

    def _get_iterator_for_epoch(self, epoch, shuffle, fix_batches_to_gpus=False, offset=0):
        # pdb.set_trace()
        if self._supports_prefetch and fix_batches_to_gpus:
            seed = self.seed + epoch + self.shard_id
        else:
            seed = self.seed + epoch
        batches = self.get_batches(shuffle=shuffle, seed=seed)

        batches = list(ShardedIterator(
            batches, self.num_shards, self.shard_id, fill_value=[]
        ))
        self.dataset.prefetch([i for s in batches for i in s])

        if offset > 0 and offset >= len(batches):
            return None

        return CountingIterator(
            torch.utils.data.DataLoader(
                self.dataset,
                collate_fn=self.collate_fn,
                batch_sampler=batches[offset:],
                num_workers=self.num_workers,
            ),
            start=offset,
        )

