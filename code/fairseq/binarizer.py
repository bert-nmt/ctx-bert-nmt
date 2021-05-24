# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import os

from fairseq.tokenizer import tokenize_line
from bert import BertTokenizer
import torch
def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line, append_eos=True, reverse_order=False,
                 offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        ntrunc = 0 if isinstance(dict, BertTokenizer) else None
        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                if isinstance(dict, BertTokenizer):
                    line = line.strip()
                    # line = '{} {} {}'.format('[CLS]', line, '[SEP]')
                    tokenizedline = dict.tokenize(line)
                    # fix token
                    tokenizedline = tokenizedline if tokenizedline[0] == '[CLS]' else (['[CLS]', ] + tokenizedline)
                    tokenizedline = tokenizedline if tokenizedline[-1] == '[SEP]' else (tokenizedline + ['[SEP]', ])
                    # truncate
                    if len(tokenizedline) > dict.max_len:
                        ntrunc += 1
                        tokenizedline = tokenizedline[:dict.max_len - 1]
                        tokenizedline.append('[SEP]')
                    words = dict.convert_tokens_to_ids(tokenizedline)
                    nwords = len(words)
                    ids = torch.IntTensor(nwords)
                    for i, word in enumerate(words):
                        ids[i] = word
                        replaced_consumer(tokenizedline[i], word)
                else:
                    ids = dict.encode_line(
                            line=line,
                            line_tokenizer=tokenize,
                            add_if_not_exist=False,
                            consumer=replaced_consumer,
                            append_eos=append_eos,
                            reverse_order=reverse_order,
                    )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced, "ntrunc": ntrunc, }

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets
