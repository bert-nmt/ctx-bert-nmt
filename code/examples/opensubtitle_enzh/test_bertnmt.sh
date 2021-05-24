# paras: checkpoint, srclng, tgtlng, beam, lenpen, subset, batch_size, extra_flag, suffix, data, reference
BASHDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if ! [[ $BASHDIR == ${FAIRSEQ}* ]]; then
    echo "Please check if FAIRSEQ is wrong."
    echo "BASHDIR $BASHDIR"
    echo "FAIRSEQ $FAIRSEQ"
    exit
fi
export PYTHONPATH=$FAIRSEQ:$PYTHONPATH

cktpath=$1
if [ ! -f $cktpath ]; then
    >&2 echo "Checkpoint $cktpath doesn't exist!"
    exit
fi

srclng=$2
tgtlng=$3
beam=${4:-5}
lenpen=${5:-1.0}
subset=${6:-test}
bs=${7:-512}

extra_flag=${8:-}
suffix=${9:-}
if [ ! -z "$extra_flag" ]; then
  if [ ! -z $suffix ]; then
    echo "Using flag: $extra_flag"
    echo "Using suffix: $suffix"
    suffix=".${suffix}"
  else
    echo "Please specify suffix when using extra flag $extra_flag"
    exit
  fi
fi

DATA_PATH=${10}
tgtlog=$(dirname ${cktpath})/${subset}/$(basename $cktpath).log$suffix
mkdir -p $(dirname ${cktpath})/${subset}
>&2 echo "Testing $srclng to $tgtlng, subset: $subset, output to $tgtlog"
>&2 echo "Using data $DATA_PATH"

if [ "$srclng" = "zh_cn" ]; then
  bert_name=bert-base-chinese
elif [ "$srclng" = "en" ]; then
  bert_name=bert-base-uncased
else
  echo "Invalid src language $src, no corresponding bert model"
  exit
fi
echo "Using BERT $bert_name"

python $FAIRSEQ/generate.py $DATA_PATH --gen-subset $subset --path $cktpath \
    --bert-model-name ${bert_name} \
    --batch-size $bs --beam $beam --lenpen $lenpen -s $srclng -t $tgtlng --remove-bpe $extra_flag > $tgtlog

python $BASHDIR/../../translation/phillyscripts/cut_generated_results.py $tgtlog $tgtlog.sys.dirty
python $BASHDIR/filter_removed.py $tgtlog.sys.dirty $refF > $tgtlog.sys

refF=${11}
if [ ! -f $refF.cleaned ]; then
  cat $refF | grep -v _removed_ > $refF.cleaned
fi
refF=$refF.cleaned

if [ $tgtlng == "zh_cn" ]; then
  pip install sacrebleu --user
  if [ ! -f $refF.detok ]; then
    cat $refF | sed "s/ //g" > $refF.detok
  fi
  cat $tgtlog.sys | sed "s/ //g" > $tgtlog.sys.detok
  python -m sacrebleu $refF.detok -l en-zh -tok zh -w 2 < $tgtlog.sys.detok
  exit
else
  if [ ! -f /tmp/mosesdecoder ]; then
    git clone https://github.com/moses-smt/mosesdecoder.git /tmp/mosesdecoder
  fi
  BLEUer="/tmp/mosesdecoder/scripts/generic/multi-bleu.perl"
  perl $BLEUer $refF < $tgtlog.sys
fi

