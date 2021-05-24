# paras: hid, src, tgt, ctx, max-tokens, batch-tokens, bert-input, pretrained

ARCH=transformer_s2_iwslt_de_en
BASHDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if ! [[ $BASHDIR == ${FAIRSEQ}* ]]; then
    echo "Please check if FAIRSEQ is wrong."
    echo "BASHDIR $BASHDIR"
    echo "FAIRSEQ $FAIRSEQ"
    exit
fi
export PYTHONPATH=$FAIRSEQ:$PYTHONPATH

hid=$1
src=${2:-de}
tgt=${3:-en}
ctx=${4:-1}
tok=${5:-6000}
dropout=0.0  # ${6:-0.1}
wd=0.0  # ${7:-0.0001}
batch_tok=${6:-"$tok"}
bert_inp=${7:-"doc"}

n_gpu=$( python -c "import torch; print(torch.cuda.device_count())" )

if [[ $tok == $batch_tok ]]; then
  update_freq=1
  extra_tok_name=""
  if [ $n_gpu != 1 ]; then
    echo "Wrong gpu setting! You use multi GPU but we only need 1 update_freq"
    exit
  fi
elif ! [[ $(( tok % batch_tok )) -eq 0 ]]; then
  echo "Wrong token setting! You set total bsz $tok and each bsz $batch_tok"
  exit
else
  _update_freq=$(( tok / batch_tok))
  if [ $n_gpu == $_update_freq ]; then
    update_freq=1
    echo "Tranin using $n_gpu GPU"
  elif [ $n_gpu != 1 ]; then
    echo "Wrong gpu setting! You use multi GPU ($n_gpu) but different from update_freq $_update_freq"
    exit
  else
    update_freq=$(( tok / batch_tok))
  fi
  extra_tok_name="_each-bsz${batch_tok}"
fi

if [ "$ctx" == 1 ]; then
  ctx_flag=""
else
  ctx_flag="_ctx${ctx}"
fi

DATA_PATH=data/iwslt14/bins/bertnmt${bert_inp}${ctx_flag}_${src}-${tgt}
echo "Training $src to $tgt, using data $DATA_PATH"

if [ "$src" = "de" ]; then
  bert_name=bert-base-german-cased
elif [ "$src" = "en" ]; then
  bert_name=bert-base-uncased
else
  echo "Invalid src language $src, no corresponding bert model"
  exit
fi
echo "Using BERT $bert_name"

# prefetch BERT
python -c "from bert import BertTokenizer; print(BertTokenizer.from_pretrained('$bert_name'))"
python -c "from bert import BertModel; print(BertModel.from_pretrained('$bert_name'))"

pretrained=$8
echo "Pretrained model: $pretrained"
if [ -z $pretrained ]; then
  echo "Pretrained model doesn't exist!"
  exit
fi

SAVEDIR=/tmp/word_embeddings_only/iwslt_${src}-${tgt}_H${hid}_tok${tok}${extra_tok_name}${ctx_flag}
echo "Save to $SAVEDIR"
if [ -f $SAVEDIR/checkpoint_last.pt ]; then
  # Resume training from checkpoint_last.pt
  echo "Resume training"
  warmup=""
else
  # Train new, require no existing savedir
  echo "Train new model"
  mkdir -p $SAVEDIR
  warmup="--warmup-from-nmt --warmup-nmt-file $pretrained --reset-dataloader --reset-lr-scheduler --reset-optimizer --reset-meters"
fi

python -u $FAIRSEQ/train.py $DATA_PATH $warmup --word-embeddings-only \
  --clip-norm 0.0 \
  -a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
  --dropout 0.3 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay $wd \
  --max-tokens $batch_tok --update-freq $update_freq \
  --criterion label_smoothed_cross_entropy --max-update 300000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --share-all-embeddings  \
  --bert-model-name $bert_name \
  --encoder-embed-dim $hid \
  --decoder-embed-dim $hid \
  --attention-dropout $dropout \
  --activation-dropout $dropout \
  --encoder-bert-dropout --encoder-bert-dropout-ratio 0.5 | tee -a $SAVEDIR/log 2>&1
