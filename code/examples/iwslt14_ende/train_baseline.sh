ARCH=transformer_iwslt_de_en
export PYTHONPATH=$FAIRSEQ:$PYTHONPATH

hid=$1
src=${2:-de}
tgt=${3:-en}
resume=${4:-"false"}

DATA_PATH=data/iwslt14/bins/fairseqbaseline_${src}-${tgt}
echo "Training $src to $tgt, using data $DATA_PATH"

SAVEDIR=/tmp/fairseq_baseline/iwslt_${src}-${tgt}_H${hid}
if [ "$resume" == "resume" ] || [ "$resume" == "true" ]; then
  # Resume training from checkpoint_last.pt
  echo "Resume training"
  if [ ! -f $SAVEDIR/checkpoint_last.pt ]; then
    echo "No $SAVEDIR/checkpoint_last.pt"
    echo "Cannot resume training"
    exit
  fi
elif [ "$resume" == "non-resume" ] || [ "$resume" == "false" ]; then
  # Train new, require no existing savedir
  echo "Train new model"
  if [ -d $SAVEDIR ] && [ ! -z "$(ls -A $SAVEDIR)" ]; then
    echo "$SAVEDIR exists and non-empty"
    exit
  fi
  echo "Save to $SAVEDIR"
  mkdir -p $SAVEDIR
else
  echo "Wrong resume setting! Only support resume / non-resume / true / false"
  echo "You set $resume"
  exit
fi

encL=6
decL=6

python -u $FAIRSEQ/train.py $DATA_PATH \
  --source-lang ${src} --target-lang ${tgt} \
  --arch $ARCH --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 5e-04 --min-lr 1e-09 \
  --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-update 300000 \
  --max-tokens 6000 \
  --save-dir $SAVEDIR \
  --seed 1 \
  --restore-file checkpoint_last.pt \
  --update-freq 1 \
  --encoder-embed-dim $hid \
  --decoder-embed-dim $hid \
  --attention-dropout 0.0 \
  --activation-dropout 0.0 \
  --distributed-no-spawn \
  --ddp-backend no_c10d \
  --encoder-layers $encL \
  --decoder-layers $decL | tee -a $SAVEDIR/log 2>&1
