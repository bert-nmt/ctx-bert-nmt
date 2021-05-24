# paras: hid, src, tgt, ctxprev, ctxnext, attn_type, dropout, bsz, update_freq, attn_ddropout, pretrained model
ARCH=transformer_han_v2
BASHDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if ! [[ $BASHDIR == ${FAIRSEQ}* ]]; then
    echo "Please check if FAIRSEQ is wrong."
    echo "BASHDIR $BASHDIR"
    echo "FAIRSEQ $FAIRSEQ"
    exit
fi
export PYTHONPATH=$FAIRSEQ:$PYTHONPATH

hid=${1:-512}  # $1
src=${2:-en}  # ${2:-en}
tgt=${3:-zh_cn}  # ${3:-ru}

ctxprev=$4
ctxnext=$5
attn_type=$6

dropout=${7:-0.1}
bsz=${8:-6000}
update_freq=${9:-2}
attn_dropout=${10:-0.1}

n_gpu=$( python -c "import torch; print(torch.cuda.device_count())" )
if (( n_gpu != 1 )); then
  if (( update_freq % n_gpu == 0 )); then
    update_freq_=$(( update_freq / n_gpu ))
    echo "Training with multi gpu, simulating update freq $update_freq; actual update freq = $update_freq_"
  else
    echo "Please set 1 gpu! Currently: $n_gpu"
    exit
  fi
else
  update_freq_=$update_freq
fi
echo $CUDA_VISIBLE_DEVICES
echo "Train with $n_gpu gpu: $bs"

exp="bertnmtsent"
DATA_PATH=data/opensubtitle_enzh/bins/${exp}_${src}-${tgt}_1wBPE_fix/
echo "Training $src to $tgt, using data $DATA_PATH"

if [ "$src" = "zh_cn" ]; then
  bert_name=bert-base-chinese
elif [ "$src" = "en" ]; then
  bert_name=bert-base-uncased
else
  echo "Invalid src language $src, no corresponding bert model"
  exit
fi
echo "Using BERT $bert_name"
python -c "from bert import BertTokenizer; print(BertTokenizer.from_pretrained('$bert_name'))"
python -c "from bert import BertModel; print(BertModel.from_pretrained('$bert_name'))"

pretrained=${11}
echo "Pretrained model: $pretrained"
if [ -z $pretrained ]; then
  echo "Pretrained model doesn't exist!"
  exit
fi

SAVEDIR=/tmp/opensubtitle_enzh/${attn_type}_${ctxprev}prev_${ctxnext}next/${src}-${tgt}_hid-${hid}_dropout-${dropout}_bsz-${bsz}_update-${update_freq}_attndropout-${attn_dropout}
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

encL=6
decL=6

python -u $FAIRSEQ/train.py $DATA_PATH $bs $warmup \
  --bert-model-name $bert_name \
  --source-lang ${src} --target-lang ${tgt} \
  --max-tokens $bsz \
  --arch $ARCH \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 5e-04 --min-lr 1e-09 \
  --dropout $dropout --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-update 300000 \
  --save-dir $SAVEDIR \
  --seed 1 \
  --restore-file checkpoint_last.pt \
  --update-freq ${update_freq_} \
  --encoder-embed-dim $hid \
  --decoder-embed-dim $hid \
  --attention-dropout ${attn_dropout} \
  --activation-dropout ${attn_dropout} \
  --distributed-no-spawn \
  --ddp-backend no_c10d \
  --task translation_han_v2 --bert-attention-type ${attn_type} \
  --n-context-prev $ctxprev --n-context-next $ctxnext \
  --encoder-layers $encL \
  --decoder-layers $decL | tee -a $SAVEDIR/log 2>&1
