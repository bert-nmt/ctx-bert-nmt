# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import pdb
import numpy as np
import torch

from . import data_utils, FairseqDataset


def construct_context_index(doc_start, n_samples, n_context_prev, n_context_next,
                            n_context_prev_preserve=None, n_context_next_preserve=None):
    # pdb.set_trace()
    docid = np.cumsum(np.array([(i in doc_start) for i in range(n_samples)], dtype=int))

    idx = np.arange(n_samples)[:, None] + np.arange(-n_context_prev, n_context_next + 1)[None, :]
    mask = (idx >= 0) * (idx < n_samples)
    idx[~mask] = 0

    ctx_docid = docid[idx]
    mask *= (docid[:, None] == ctx_docid)

    if n_context_prev_preserve is not None or n_context_next_preserve is not None:
        idx_self = n_context_prev
        if n_context_prev_preserve is not None:
            assert n_context_prev_preserve <= n_context_prev
            mask[:, :idx_self - n_context_prev_preserve] = 0
        if n_context_next_preserve is not None:
            # pdb.set_trace()
            assert n_context_next_preserve <= n_context_next
            mask[:, idx_self + n_context_next_preserve + 1:] = 0

    return torch.LongTensor(idx), torch.ByteTensor(mask.astype(np.int))


class LanguagePairDocumentFlexibleCtxDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
            self, doc_index, src, src_sizes, src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            srcbert=None, srcbert_sizes=None, berttokenizer=None,
            n_context_prev=1, n_context_next=0, full_context=True,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
            n_context_prev_preserve=None,
            n_context_next_preserve=None,
    ):
        # pdb.set_trace()
        self.doc_index = doc_index
        self.n_context_prev = n_context_prev
        self.n_context_next = n_context_next
        self.full_context = full_context
        self.ctx_idx, self.ctx_mask = construct_context_index(self.doc_index, len(src), n_context_prev, n_context_next,
                                                              n_context_prev_preserve=n_context_prev_preserve,
                                                              n_context_next_preserve=n_context_next_preserve)

        # self.start_of_doc = [(i in self.doc_index) for i in range(len(src))]
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.srcbert = srcbert
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.srcbert_sizes = np.array(srcbert_sizes) if srcbert_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.berttokenizer = berttokenizer
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        # pdb.set_trace()
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        src_bert_item = self.srcbert[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'source_bert': src_bert_item,
            'ctx_idx': self.ctx_idx[index],
            'ctx_mask': self.ctx_mask[index],
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        # pdb.set_trace()

        pad_idx = self.src_dict.pad()
        eos_idx = self.src_dict.eos()
        bert_pad_idx = self.berttokenizer.pad()
        left_pad_source = self.left_pad_source
        left_pad_target = self.left_pad_target
        input_feeding = self.input_feeding

        if len(samples) == 0:
            return {}

        def merge(key, left_pad, move_eos_to_beginning=False, bert_input=False):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx if not bert_input else bert_pad_idx, eos_idx, left_pad, move_eos_to_beginning
            )

        # pdb.set_trace()
        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=left_pad_source)
        ctx_idx = torch.stack([s['ctx_idx'] for s in samples], dim=1)
        ctx_mask = torch.stack([s['ctx_mask'] for s in samples], dim=1)

        # process ctx idx
        def torch_in(x, y):
            shape_x = x.shape
            z = torch.any(x.reshape(-1)[:, None] == y.reshape(-1)[None, :], dim=1)
            return z.reshape(x.shape)

        # pdb.set_trace()
        if self.full_context:
            bert_idx = sorted(set([int(i) for i in ctx_idx[ctx_mask].reshape(-1)]))
        else:
            ctx_mask *= torch_in(ctx_idx, id)
            bert_idx = [int(i) for i in id]
        ctx_idx[~ctx_mask] = min(bert_idx)
        ctx_idx = torch.LongTensor(
            [bert_idx.index(i) for i in ctx_idx.reshape(-1)]
        ).reshape(ctx_idx.shape)

        # load bert
        bert_tokens = data_utils.collate_tokens(
            [self.srcbert[i] for i in bert_idx],
            bert_pad_idx, eos_idx, left_pad_source, False
        )

        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        # src_bert_tokens = src_bert_tokens.index_select(0, sort_order)
        ctx_idx = ctx_idx.index_select(1, sort_order)
        ctx_mask = ctx_mask.index_select(1, sort_order)

        prev_output_tokens = None
        target = None
        if samples[0].get('target', None) is not None:
            target = merge('target', left_pad=left_pad_target)
            target = target.index_select(0, sort_order)
            ntokens = sum(len(s['target']) for s in samples)

            if input_feeding:
                # we create a shifted version of targets for feeding the
                # previous output token(s) into the next decoder step
                prev_output_tokens = merge(
                    'target',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
                prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        else:
            ntokens = sum(len(s['source']) for s in samples)

        # start_of_doc = torch.ByteTensor([s['start_of_doc'] for s in samples])

        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'bert_encoder_input': {
                    'bert_tokens': bert_tokens,  # src_bert_tokens,
                    'ctx_idx': ctx_idx,
                    'ctx_mask': ctx_mask,
                    # 'bert_idx': torch.LongTensor(bert_idx),
                },
            },
            'target': target,
        }
        if prev_output_tokens is not None:
            batch['net_input']['prev_output_tokens'] = prev_output_tokens
        return batch

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        a = max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return max(a, max([self.srcbert_sizes[i] for i in self.ctx_idx[index]]))

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        self.srcbert.prefetch(indices)
