# paras: checkpoint, srclng, tgtlng, beam, lenpen, subset, batch_size, extra_flag, suffix, data, reference
BASHDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"  # find bash script dir to call python script

if ! [[ $BASHDIR == ${FAIRSEQ}* ]]; then
    >&2 echo "Please check if FAIRSEQ is wrong."
    >&2 echo "BASHDIR $BASHDIR"
    >&2 echo "FAIRSEQ $FAIRSEQ"
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
    >&2 echo "Using flag: $extra_flag"
    >&2 echo "Using suffix: $suffix"
    suffix=".${suffix}"
  else
    >&2 echo "Please specify suffix when using extra flag $extra_flag"
    exit
  fi
fi

DATA_PATH=${10}
tgtlog=$(dirname ${cktpath})/${subset}/$(basename $cktpath).log$suffix
mkdir -p $(dirname ${cktpath})/${subset}
>&2 echo "Testing $srclng to $tgtlng, subset: $subset, output to $tgtlog"
>&2 echo "Using data $DATA_PATH"

if [ "$srclng" = "de" ]; then
  bert_name=bert-base-german-cased
elif [ "$srclng" = "en" ]; then
  bert_name=bert-base-uncased
else
  >&2 echo "Invalid src language $srclng, no corresponding bert model"
  exit
fi
>&2 echo "Using BERT $bert_name"

python $FAIRSEQ/generate.py $DATA_PATH --gen-subset $subset --path $cktpath \
    --bert-model-name $bert_name \
    --batch-size $bs --beam $beam --lenpen $lenpen -s $srclng -t $tgtlng --remove-bpe $extra_flag > $tgtlog
python $BASHDIR/cut_generated_results.py $tgtlog $tgtlog.sys

refF=${11}
if [ ! -f /tmp/mosesdecoder ]; then
  git clone https://github.com/moses-smt/mosesdecoder.git /tmp/mosesdecoder
fi
BLEUer="/tmp/mosesdecoder/scripts/generic/multi-bleu.perl"
perl $BLEUer $refF < $tgtlog.sys

