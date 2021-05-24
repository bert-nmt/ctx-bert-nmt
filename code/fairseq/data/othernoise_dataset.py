import pdb
import torch

from . import FairseqDataset


class OtherNoiseDataset(FairseqDataset):

    def __init__(self, dataset, noise_paras=None):
        super(OtherNoiseDataset, self).__init__()
        self.dataset = dataset
        self.noising = SingleSentNoising(**noise_paras) if noise_paras is not None else None
        self._data = dict()

    def _noise_sample(self, x):
        return self.noising.noising(x) if self.noising is not None else x

    def __getitem__(self, index):
        return self._noise_sample(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    @property
    def sizes(self):
        return self.dataset.sizes

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    def prefetch(self, indices):
        # pdb.set_trace()
        self.dataset.prefetch(indices)


class SingleSentNoising:
    def __init__(self, dictionary, masking_prob):
        self.dictionary = dictionary
        self.masking_prob = masking_prob

    def mask(self, x):
        has_eos = x[-1] == self.dictionary.eos()
        # pdb.set_trace()
        if has_eos:
            keep = torch.rand((len(x) - 1, ))
            keep = torch.cat([keep, torch.FloatTensor([1.0, ])], dim=0)
        else:
            keep = torch.rand((len(x), ))
        keep = (keep >= self.masking_prob).to(x.device).long()
        return x * keep + self.dictionary.unk() * (1 - keep)

    def noising(self, x):
        if self.masking_prob > 0:
            x = self.mask(x)
        return x

