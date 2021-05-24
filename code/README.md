This project is modified from [fairseq](https://github.com/pytorch/fairseq/). For the original README, please check [README.fairseq.md](README.fairseq.md).

# Experiments

For all scripts, you should set `FAIRSEQ` environment variable, which indicates the root path of the fairseq package. To run the baseline models, you should clone fairseq repo from [fairseq](https://github.com/pytorch/fairseq/) (we use fairseq v0.8.0). To run our model, you should set `FAIRSEQ` as the root path of this code.

* To run experiments for IWSLT'14 En-De, please refer to [examples/iwslt14_ende](examples/iwslt14_ende)
* To run experiments for OpenSubtitle'18 Zh-En, please refer to [examples/opensubtitle_enzh](examples/opensubtitle_enzh)

# Code

BERT code is modified from [huggingface transformers](https://github.com/huggingface/transformers) and the code is placed in `bert`.

C-mode translation task is implemented in `fairseq/tasks/translation.py`, and the model is implemented in `fairseq/models/transformer.py`.

F-mode and H-mode translation task is implemented in `fairseq/tasks/translation_han_v2.py`, the model is implemented in `fairseq/models/transformer_han_v2.py`, and the attention modules are implemented in `fairseq/modules/han_attention.py`.