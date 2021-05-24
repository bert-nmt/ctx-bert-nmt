# paras: path, src, tgt, workers
path=$1
TEXT=$path/iwslt14.tokenized.de-en
src=${2:-de}
tgt=${3:-en}
w=${4:-10}

ppath=$PYTHONPATH

# fairseqbaseline
export PYTHONPATH=$FAIRSEQ:$ppath
destdir=$path/bins/fairseqbaseline_${src}-${tgt}
echo "fairseqbaseline: dest dir $destdir"
if [ -d $destdir ] && [ ! -z "$(ls -A $destdir)" ]; then
  echo "$destdir exists and non-empty; quit"
else
  mkdir $destdir -p
  python $FAIRSEQ/preprocess.py --source-lang $src --target-lang $tgt \
    --joined-dictionary \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $destdir --workers $w
fi
echo