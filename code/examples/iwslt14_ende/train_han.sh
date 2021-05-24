# paras: hid, src, tgt, attn-type, n_ctx, max-tokens, max-bert-tokens, batch-tokens, pretrained

ARCH=transformer_han_v2_iwslt_de_en
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
attn_type=${4:-flat}
nctx=${5:-1}
tok=${6:-6000}
btok=${7}
dropout=0.0  # ${6:-0.1}
wd=0.0  # ${7:-0.0001}
batch_tok=${8:-"$tok"}

if [[ $tok == $batch_tok ]]; then
  update_freq=1
  extra_tok_name=""
elif ! [[ $(( tok % batch_tok )) -eq 0 ]]; then
  echo "Wrong token setting! You set total bsz $tok and each bsz $batch_tok"
  exit
else
  update_freq=$(( tok / batch_tok))
  extra_tok_name="_each-bsz${batch_tok}"
fi

if [ ! -z $btok ]; then
  flag="--max-bert-tokens $btok "
  extra_tok_name="${extra_tok_name}_btok$btok"
fi

DATA_PATH=data/iwslt14/bins/bertnmtsent_${src}-${tgt}
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

pretrained=$9
echo "Pretrained model: $pretrained"
if [ -z $pretrained ]; then
  echo "Pretrained model doesn't exist!"
  exit
fi

SAVEDIR=/tmp/bertnmtdoc_han/iwslt_${nctx}ctx_attn-${attn_type}_${src}-${tgt}_H${hid}_tok${tok}${extra_tok_name}
echo "Save to $SAVEDIR"
if [ -f $SAVEDIR/checkpoint_last.pt ]; then
  # Resume training from checkpoint_last.pt
  echo "Resume training"
  flag="$flag"
else
  # Train new, require no existing savedir
  echo "Train new model"
  mkdir -p $SAVEDIR
  flag="$flag --warmup-from-nmt --warmup-nmt-file $pretrained --reset-dataloader --reset-lr-scheduler --reset-optimizer --reset-meters"
fi

python -u $FAIRSEQ/train.py $DATA_PATH $flag \
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
  --task translation_han_v2 --bert-attention-type ${attn_type} --n-context $nctx \
  --encoder-bert-dropout --encoder-bert-dropout-ratio 0.5 | tee -a $SAVEDIR/log 2>&1

