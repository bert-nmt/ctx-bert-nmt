# paras: hid, src, tgt, dropout, bsz, update_freq, attn_ddropout
ARCH=transformer  # _iwslt_de_en; base model
export PYTHONPATH=$FAIRSEQ:$PYTHONPATH

hid=${1:-512}  # $1
src=${2:-en}  # ${2:-en}
tgt=${3:-zh_cn}  # ${3:-ru}

dropout=${4:-0.1}
bsz=${5:-6000}
update_freq=${6:-2}
attn_dropout=${7:-0.1}

n_gpu=$( python -c "import torch; print(torch.cuda.device_count())" )
if (( n_gpu != 1 )); then
  echo "Please set 1 gpu! Currently: $n_gpu"
  exit
fi
echo $CUDA_VISIBLE_DEVICES
echo "Train with $n_gpu gpu: $bs"

DATA_PATH=data/opensubtitle_enzh/bins/fairseqbaseline_en-zh_cn/
echo "Training $src to $tgt, using data $DATA_PATH"

SAVEDIR=/tmp/opensubtitle_enzh/fairseqbaseline/${src}-${tgt}_hid-${hid}_dropout-${dropout}_bsz-${bsz}_update-${update_freq}_attndropout-${attn_dropout}
echo "Save to $SAVEDIR"
if [ -f $SAVEDIR/checkpoint_last.pt ]; then
  # Resume training from checkpoint_last.pt
  echo "Resume training"
else
  echo "Train new model"
  mkdir -p $SAVEDIR
fi

encL=6
decL=6

python -u $FAIRSEQ/train.py $DATA_PATH $bs \
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
  --update-freq ${update_freq} \
  --encoder-embed-dim $hid \
  --decoder-embed-dim $hid \
  --attention-dropout ${attn_dropout} \
  --activation-dropout ${attn_dropout} \
  --distributed-no-spawn \
  --ddp-backend no_c10d \
  --encoder-layers $encL \
  --decoder-layers $decL | tee -a $SAVEDIR/log 2>&1
