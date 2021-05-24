# paras: path, src, tgt, workers
path=$1
TEXT=$path
src=${2:-en}
tgt=${3:-zh_cn}
w=${4:-10}

# fairseqbaseline
export PYTHONPATH=$FAIRSEQ:$PYTHONPATH
destdir=$path/bins/fairseqbaseline_${src}-${tgt}
echo "fairseqbaseline: test dir $TEXT, dest dir $destdir"
if [ -d $destdir ] && [ ! -z "$(ls -A $destdir)" ]; then
  echo "$destdir exists and non-empty; quit"
else
  mkdir $destdir -p
  python $FAIRSEQ/preprocess.py --source-lang $src --target-lang $tgt \
    --joined-dictionary \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $destdir --workers $w
fi

