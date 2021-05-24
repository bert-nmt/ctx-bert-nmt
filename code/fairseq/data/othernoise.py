import torch
import numpy as np
import os
import io
from fairseq.data import data_utils


class WordNoising(object):
    """Generate a noisy version of a sentence, without changing words themselves."""
    def __init__(self, dictionary,):
        self.dictionary = dictionary


    def noising(self, x, lengths, noising_prob=0.0, target=False):
        raise NotImplementedError()

    def get_word_idx(self, x):
        # x: (T x B)
        if (x.size(0) == 1 and x.size(1) == 1):
            # Special case when we only have one word in x. If x = [[N]],
            # bpe_end is a scalar (bool) instead of a 2-dim array of bools,
            # which makes the sum operation below fail.
            return np.array([[0]])
        tmp = np.ones(x.shape)
        tmp = tmp.cumsum(0)
        word_idx = tmp - 1
        word_idx = np.int64(word_idx)
        return word_idx


class WordDropout(WordNoising):
    """Randomly drop input words. If not passing blank_idx (default is None),
    then dropped words will be removed. Otherwise, it will be replaced by the
    blank_idx."""

    def __init__(self, dictionary):
        super().__init__(dictionary, )

    def noising(self, x, lengths, dropout_prob=0.1, blank_idx=None, target=False):
        # x: (T x B), lengths: B
        if dropout_prob == 0:
            return x, lengths
        device = x.device
        assert 0 < dropout_prob < 1

        # be sure to drop entire words
        sentences = []
        modified_lengths = []
        if not target:
            # left pad source
            for i in range(lengths.size(0)):
                num_words = x.size(0)
                has_eos = x[-1, i] == self.dictionary.eos()
                if has_eos:  # has eos?
                    keep = np.random.rand(num_words - 1) >= dropout_prob
                    keep = np.append(keep, [True])  # keep EOS symbol
                else:
                    keep = np.random.rand(num_words) >= dropout_prob

                words = x[-lengths[i]:, i].tolist()

                # TODO: speed up the following loop
                # drop words from the input according to keep
                pad_num = num_words - lengths[i]
                new_s = [
                    w if keep[j + pad_num] else blank_idx
                    for j, w in enumerate(words)
                ]
                new_s = [w for w in new_s if w is not None]
                # we need to have at least one word in the sentence (more than the
                # start / end sentence symbols)
                if len(new_s) <= 1:
                    # insert at beginning in case the only token left is EOS
                    # EOS should be at end of list.
                    # -1 means that eos should be excluded
                    new_s.insert(0, words[np.random.randint(0, len(words) - 1)])
                assert len(new_s) >= 1 and (
                    not has_eos  # Either don't have EOS at end or last token is EOS
                    or (len(new_s) >= 2 and new_s[-1] == self.dictionary.eos())
                ), "New sentence is invalid."
                sentences.append(new_s)
                modified_lengths.append(len(new_s))
            modified_lengths = torch.LongTensor(modified_lengths)
            modified_x = torch.LongTensor(
                modified_lengths.max(),
                modified_lengths.size(0)
            ).fill_(self.dictionary.pad())
            for i in range(modified_lengths.size(0)):
                modified_x[-modified_lengths[i]:, i].copy_(torch.LongTensor(sentences[i]))
        else:
            if blank_idx is None:
                return x, lengths
            # right pad the target sentence
            for i in range(lengths.size(0)):
                num_words = x.size(0)
                has_eos = x[0, i] == self.dictionary.eos()
                if has_eos:  # has eos?
                    keep = np.random.rand(num_words - 1) >= dropout_prob
                    keep = np.append( [True], keep)  # keep EOS symbol
                else:
                    keep = np.random.rand(num_words) >= dropout_prob

                words = x[:lengths[i], i].tolist()

                # TODO: speed up the following loop
                # drop words from the input according to keep
                new_s = [
                    w if keep[j ] else blank_idx
                    for j, w in enumerate(words)
                ]
                new_s = [w for w in new_s if w is not None]
                # we need to have at least one word in the sentence (more than the
                # start / end sentence symbols)
                # if len(new_s) <= 1:
                #     # insert at beginning in case the only token left is EOS
                #     # EOS should be at end of list.
                #     # -1 means that eos should be excluded
                #     new_s.insert(1, words[np.random.randint(1, len(words) )])
                # assert len(new_s) >= 1 and (
                #         not has_eos  # Either don't have EOS at end or last token is EOS
                #         or (len(new_s) >= 2 and new_s[0] == self.dictionary.eos())
                # ), "New sentence is invalid."
                sentences.append(new_s)
                modified_lengths.append(len(new_s))
        # re-construct input
            modified_lengths = torch.LongTensor(modified_lengths)
            modified_x = torch.LongTensor(
                modified_lengths.max(),
                modified_lengths.size(0)
            ).fill_(self.dictionary.pad())
            for i in range(modified_lengths.size(0)):
                modified_x[:modified_lengths[i], i].copy_(torch.LongTensor(sentences[i]))

        return modified_x.to(device), modified_lengths.to(device)




class WordShuffle(WordNoising):
    """Shuffle words by no more than k positions."""

    def __init__(self, dictionary,):
        super().__init__(dictionary, )

    def noising(self, x, lengths, max_shuffle_distance=3, target=False):
        # x: (T x B), lengths: B
        if max_shuffle_distance == 0:
            return x, lengths
        device = x.device
        # max_shuffle_distance < 1 will return the same sequence
        assert max_shuffle_distance > 1

        # define noise word scores
        noise = np.random.uniform(
            0,
            max_shuffle_distance,
            size=(x.size(0), x.size(1)),
        )

        word_idx = self.get_word_idx(x)
        x2 = x.clone()
        if target:
            return x, lengths
            # right pad the target sentence
            # for i in range(lengths.size(0)):
            #     length_with_eos = lengths[i]
            #     if x[0, i] == self.dictionary.eos():
            #         idx_satrt = 1
            #     else:
            #         idx_satrt = 0
            #     # generate a random permutation
            #     scores = word_idx[idx_satrt:length_with_eos, i] + noise[word_idx[idx_satrt:length_with_eos, i], i]
            #     permutation = scores.argsort()
            #     # shuffle words
            #     x2[idx_satrt:length_with_eos, i].copy_(
            #         x2[idx_satrt:length_with_eos, i][torch.from_numpy(permutation)]
            #     )
        else:
            # left pad the source sentence
            for i in range(lengths.size(0)):
                length_with_eos = lengths[i]

                # generate a random permutation
                scores = word_idx[-length_with_eos: -1, i] + noise[word_idx[-length_with_eos: -1, i], i]
                permutation = scores.argsort()
                # shuffle words
                x2[-length_with_eos: -1, i].copy_(
                    x2[-length_with_eos: -1, i][torch.from_numpy(permutation)]
                )
        return x2.to(device), lengths

class WordSmooth(WordNoising):
    """Smooth word inputs as in iclr paper"""

    def __init__(self, dictionary, args):
        super().__init__(dictionary)
        dictdir = args.data[0]
        source_lang = args.source_lang
        target_lang = args.target_lang
        self.srcproblist = self.readdictfreq(dictdir, source_lang)
        self.tgtproblist = self.readdictfreq(dictdir, target_lang)


    def readdictfreq(self, dictdir, lang):
        fn = os.path.join(dictdir, 'dict.'+lang+'.txt')
        assert os.path.exists(fn), fn + ' does not exist'
        problist = []
        with io.open(fn, 'r', encoding='utf8', newline='\n') as src:
            for line in src:
                word, num = line.strip().split()
                problist.append(int(num))
        total = sum(problist)
        problist = [float(x)/total for x in problist]
        for i in range(4):
            problist.insert(0, 0.)
        return problist

    def noising(self, x, lengths, noising_prob=0.0, target=False ):
        problist = self.tgtproblist if target else self.srcproblist
        noise_x = np.random.choice(len(problist), x.shape, p=problist)
        noise_x = torch.from_numpy(noise_x).to(x.device).type(x.type())
        keep1 = (x == self.dictionary.pad())
        keep2 = (x == self.dictionary.unk())
        keep3 = (x == self.dictionary.eos())
        keep4 = torch.from_numpy(np.random.rand(*x.shape)).to(x.device) > noising_prob
        keep = (keep1 + keep2 + keep3 + keep4) > 0
        x = x.masked_fill(1 - keep, 0) + noise_x.masked_fill(keep, 0)
        return x, lengths

class UnsupervisedMTNoising(WordNoising):
    """
    Implements the default configuration for noising in UnsupervisedMT
    (github.com/facebookresearch/UnsupervisedMT)
    """
    def __init__(
        self,
        dictionary,
        max_word_shuffle_distance,
        word_dropout_prob,
        word_blanking_prob,
        word_smooth_prob,
        args
    ):
        super().__init__(dictionary)
        assert args.left_pad_source
        assert not args.left_pad_target
        self.max_word_shuffle_distance = max_word_shuffle_distance
        self.word_dropout_prob = word_dropout_prob
        self.word_blanking_prob = word_blanking_prob
        self.word_smooth_prob = word_smooth_prob
        if self.word_dropout_prob > 0 or self.word_blanking_prob > 0:
            self.word_dropout = WordDropout(
                dictionary=dictionary,
            )
        if self.max_word_shuffle_distance > 0:
            self.word_shuffle = WordShuffle(
                dictionary=dictionary,
            )
        if self.word_smooth_prob > 0:
            self.word_smooth  = WordSmooth(dictionary, args)

    def noising(self, x, lengths, target=False):
        # x: (T, B); lengths: (B, )
        # 1. Word Shuffle
        if self.max_word_shuffle_distance > 0:
            x, lengths = self.word_shuffle.noising(
                x=x,
                lengths=lengths,
                max_shuffle_distance=self.max_word_shuffle_distance,
                target=target
            )
        # 2. Word Dropout
        if self.word_dropout_prob > 0:
            x, lengths = self.word_dropout.noising(
                x=x,
                lengths=lengths,
                dropout_prob=self.word_dropout_prob,
                target=target
            )
        # 3. Word Blanking
        if self.word_blanking_prob > 0:
            x, lengths = self.word_dropout.noising(
                x=x,
                lengths=lengths,
                dropout_prob=self.word_blanking_prob,
                blank_idx=self.dictionary.unk(),
                target=target
            )
        # 4. word Smoothing
        if self.word_smooth_prob > 0:
            x, lengths = self.word_smooth.noising(
                x= x,
                lengths=lengths,
                noising_prob=self.word_smooth_prob,
                target=target
            )

        return x, lengths


if __name__ == "__main__":
    from fairseq.data import Dictionary, data_utils
    import sys
    # TODO: this code assumes src left pad & tgt right pad

    d = sys.argv[1]
    d = Dictionary.load(d)
    # noising
    noising = UnsupervisedMTNoising(d, 0, 0, 0.3, 0, None)  # use blank only;
                                                            # doesn't need args as long as don't use smooth

    # input, encode, noise
    while True:
        print("Input 3 sents to form batch")
        lines = [input() for _ in range(3)]
        tokens = [d.encode_line(line) for line in lines]
        print(tokens)
        # batch
        length = torch.LongTensor([len(x) for x in tokens])
        leftpad_tokens = data_utils.collate_tokens(tokens, d.pad(), d.eos(), True, False)
        rightpad_tokens = data_utils.collate_tokens(tokens, d.pad(), d.eos(), False, False)
        # noise
        print(noising.noising(leftpad_tokens, length))
        print(noising.noising(rightpad_tokens, length, target=True))

