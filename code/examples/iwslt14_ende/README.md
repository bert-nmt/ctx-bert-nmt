First, prepare data using `prepare/prepare_iwslt14.sh`

# Baseline

1. **Binarize the data** using `prepare/binarize_baseline.sh`. The arguments are: the root path of the data, source language (en or zh_cn), target language (en or zh_cn), and the number of workers for binarizing.
2. **Train** using `train_baseline.sh`
3. **Test** using `test_baseline.sh`

# C-mode, BERT-NMT

1. **Prepare BERT input** using `prepare/makedataforbert.sh`. (Note: you should call this script twice, with argument "en" and "zh_cn" respectively.)
2. For C-mode, **prepare document-level input** using `prepare/makedatafordoc_xxx.py` (use different scripts according to the contextual information you would like to use).
3. **Binarize the data** using `prepare/binarize_bertnmtdoc.sh`.
4. **Train** using `train_bertnmt.sh`
5. **Test** using `test_bertnmt.sh`

## Ablations

Scripts for running ablation studies are also included.

1. To run C-mode with irrelated context, use `prepare/makedatafordoc_fixed.py` and `prepare/makedatafordoc_random.py`
2. To run C-mode with word embeddings as contextual encoder, use `train_bertnmt.word_embeddings_only.sh`

# F-mode, H-mode

1. Prepare input and binarize the data, as in BERT-NMT.
2. **Generate index file** using `python prepare_index.py INP OUP`, where INP is the input directory (the data root path), OUP is the output directory (the path to the binarized data of BERT-NMT).
3. **Train** using `train_han.sh`
4. **Test** using `test_han.sh`

# Notes

* All experiments require 1 GPU.
* Before training, please re-check the data path is correct on your computer.
* To learn the usage of the scripts, please check the first line of the bash scripts, and call `python xxx.py --help` for python scripts.
* For C-mode, F-mode and H-mode, you need to specify a pretrained sentence-level NMT checkpoint. We recommend using the `checkpoint_best.pt` of baseline experiment.