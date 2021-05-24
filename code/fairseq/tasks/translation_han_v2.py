# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import pdb
import itertools
import numpy as np
import os
from bert import BertTokenizer
from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDocumentFlexibleCtxDataset,
    DocumentEpochBatchIterator,
)

from . import FairseqTask, register_task


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    n_context_prev, n_context_next, n_context_prev_preserve, n_context_next_preserve,
    full_context,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions, max_target_positions, bert_model_name
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    srcbert_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
            bertprefix = os.path.join(data_path, '{}.bert.doc.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
            bertprefix =  os.path.join(data_path, '{}.bert.doc.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_datasets.append(indexed_dataset.make_dataset(prefix + src, impl=dataset_impl,
                                                         fix_lua_indexing=True, dictionary=src_dict))
        tgt_datasets.append(indexed_dataset.make_dataset(prefix + tgt, impl=dataset_impl,
                                                         fix_lua_indexing=True, dictionary=tgt_dict))
        srcbert_datasets.append(indexed_dataset.make_dataset(bertprefix + src, impl=dataset_impl,
                                                         fix_lua_indexing=True, ))

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        srcbert_datasets = srcbert_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    # pdb.set_trace()
    doc_index = []
    with open(os.path.join(data_path, '{}.doc_index'.format(split)), encoding='utf-8') as f:
        for line in f:
            doc_index.append(int(line.strip()))
    doc_index = np.array(doc_index, dtype=np.int)

    berttokenizer = BertTokenizer.from_pretrained(bert_model_name)
    return LanguagePairDocumentFlexibleCtxDataset(
        doc_index,
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        srcbert_datasets, srcbert_datasets.sizes, berttokenizer,
        n_context_prev, n_context_next, full_context,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        n_context_prev_preserve=n_context_prev_preserve,
        n_context_next_preserve=n_context_next_preserve,
    )


@register_task('translation_han_v2')
class TranslationHANv2Task(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--bert-model-name', default='bert-base-uncased', type=str)
        parser.add_argument('--encoder-ratio', default=1., type=float)
        parser.add_argument('--bert-ratio', default=1., type=float)
        parser.add_argument('--finetune-bert', action='store_true')
        parser.add_argument('--mask-cls-sep', action='store_true')
        parser.add_argument('--warmup-from-nmt', action='store_true', )
        parser.add_argument('--warmup-nmt-file', default='checkpoint_nmt.pt', )
        parser.add_argument('--bert-gates', default=[1, 1, 1, 1, 1, 1], nargs='+', type=int)
        parser.add_argument('--bert-first', action='store_false', )
        parser.add_argument('--encoder-bert-dropout', action='store_true',)
        parser.add_argument('--encoder-bert-dropout-ratio', default=0.25, type=float)
        parser.add_argument('--bert-output-layer', default=-1, type=int)
        parser.add_argument('--encoder-bert-mixup', action='store_true')
        # fmt: on
        parser.add_argument('--n-context-prev', default=1, type=int)
        parser.add_argument('--n-context-next', default=0, type=int)
        parser.add_argument('--not-full-context', default=False, action='store_true',
                            help="don't use full context as input when training; same as previous code")

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.bert_model_name = args.bert_model_name

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            n_context_prev=self.args.n_context_prev,
            n_context_next=self.args.n_context_next,
            full_context=not self.args.not_full_context,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            bert_model_name = self.bert_model_name,
            n_context_prev_preserve=getattr(self.args, "n_context_prev_preserve", None),
            n_context_next_preserve=getattr(self.args, "n_context_next_preserve", None),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, srcbert, srcbert_sizes, berttokenizer):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary, srcbert=srcbert, srcbert_sizes=srcbert_sizes, berttokenizer=berttokenizer)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
