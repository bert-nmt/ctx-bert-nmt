#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
src=en
tgt=zh_cn
lang=${src}-${tgt}

if [[ $src != zh* ]] && [[ $tgt != zh* ]]; then
  echo "Must at least one zh related language"
  exit
fi

if [ ! -f text/OpenSubtitles.$lang.$src ] || [ ! -f text/OpenSubtitles.$lang.$tgt ] || [ ! -f text/OpenSubtitles.$lang.ids ]; then
  echo "Imcomplete file"
  exit
fi

BASHDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"  # find bash script dir to call python script

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo "Cloning fastBPE from GitHub repository..."
git clone https://github.com/glample/fastBPE
if [ ! -d fastBPE/fast ]; then
  cd fastBPE
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
  cd ..
fi

pip install jieba --user

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
# CLEAN=$SCRIPTS/training/clean-corpus-n.perl
FASTBPE=fastBPE/fast
BPE_TOKENS=10000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

# clean corpus by rules
mkdir -p tmp
python $BASHDIR/clean_corpus.py text/OpenSubtitles.$lang tmp/$lang.cleaned.tmp
# tokenize
for l in $src $tgt; do
    inp=tmp/$lang.cleaned.tmp.$l
    oup=tmp/$lang.tok.$l
    echo "Tokenize $l  $inp  ==>  $oup"
    if [[ $l = zh* ]]; then
	cat $inp | python $BASHDIR/zh_jieba.py > $oup
    else
	cat $inp | perl $TOKENIZER -threads 8 -l $l > $oup
    fi
    sed -i 's/^_ removed _$/_removed_/g' $oup
done
# clean corpus by length
python $BASHDIR/clean_n_tokenized.py tmp/$lang.tok tmp/$lang.cleaned \
    --en-by-zh-ratio 4.0 --zh-by-en-ratio 2.5 --min-len 1 --max-len 175

# lower case data
for l in $src $tgt; do
    echo "Lower case:  tmp/$lang.cleaned.$l  ==>  tmp/$lang.lc.$l"
    perl $LC < tmp/$lang.cleaned.$l > tmp/$lang.lc.$l
done

# split documents into tmp/docs/
python $BASHDIR/split_documents.py  tmp/$lang.lc.$src  tmp/$lang.lc.$tgt  text/OpenSubtitles.$lang.ids
# filter documents
python $BASHDIR/filter_docs.py  tmp/docs/$lang.lc.$src.doc_  --ratio 0.2 > tmp/docs/excluded.ids
# select train test valid
python $BASHDIR/sample_docs.py  tmp/docs/$lang.lc  --src  $src  --tgt  $tgt \
    --exclude  tmp/docs/excluded.ids \
    --n 2000000 --out tmp/train
python $BASHDIR/sample_docs.py  tmp/docs/$lang.lc  --src  $src  --tgt  $tgt \
    --exclude  tmp/docs/excluded.ids  tmp/train.ids \
    --n 10000 --out tmp/valid
python $BASHDIR/sample_docs.py  tmp/docs/$lang.lc  --src  $src  --tgt  $tgt \
    --exclude  tmp/docs/excluded.ids  tmp/train.ids  tmp/valid.ids \
    --n 10000 --out tmp/test
# remove start of doc
mkdir -p final_1wBPE_fix
for x in train valid test; do
    for l in $src $tgt; do
	cat tmp/$x.$l.with_startofdoc | grep -v startofdocumentplaceholder > tmp/$x.$l
        cp tmp/$x.$l final_1wBPE_fix/$x.$l.tok
    done
done

# BPE respectively
for l in $src $tgt; do
    BPE_CODE=final_1wBPE_fix/code.$l

    echo "learn_bpe.py on tmp/train.$l..."
    $FASTBPE learnbpe $BPE_TOKENS tmp/train.$l > $BPE_CODE

    for f in train.$l valid.$l test.$l; do
        echo "apply_bpe.py to ${f}..."
        $FASTBPE applybpe final_1wBPE_fix/$f tmp/$f $BPE_CODE
    done
done

