# paras: path, ctx, src, tgt, workers
path=$1
TEXT=$path
ctx=$2
src=${3:-en}
tgt=${4:-zh_cn}
w=${5:-10}

# bertnmtdoc
export PYTHONPATH=$FAIRSEQ:$PYTHONPATH
if [ "$src" = "zh_cn" ]; then
  bert_name=bert-base-chinese
elif [ "$src" = "en" ]; then
  bert_name=bert-base-uncased
else
  echo "Invalid src language $src, no corresponding bert model"
  exit
fi
echo "Using BERT $bert_name"
echo

if [ "$ctx" == "sent" ]; then
  # sent level
  destdir=$path/bins/bertnmtsent_${src}-${tgt}
  echo "bertnmt sent-level: dest dir $destdir"
  bert_input=sent
  prefsuf=""
else
  destdir=$path/bins/bertnmtdoc${ctx}_${src}-${tgt}
  echo "bertnmt doc-level: dest dir $destdir"
  bert_input=doc
  prefsuf=$ctx
fi

for x in train valid test; do
  for l in $src $tgt; do
    if [ ! -f $TEXT/${x}${ctx}.$l ]; then
      echo "Copy $TEXT/${x}${ctx}.$l"
      cp  $TEXT/${x}.$l  $TEXT/${x}${ctx}.$l
    fi
  done
done

if [ -d $destdir ] && [ ! -z "$(ls -A $destdir)" ]; then
  echo "$destdir exists and non-empty; quit"
else
  mkdir $destdir -p
  dict=$path/bins/fairseqbaseline_en-zh_cn
  python $FAIRSEQ/preprocess.py --source-lang $src --target-lang $tgt --bert-input ${bert_input} \
    --srcdict $dict/dict.$src.txt --tgtdict $dict/dict.$tgt.txt \
    --trainpref $TEXT/train${prefsuf} --validpref $TEXT/valid${prefsuf} --testpref $TEXT/test${prefsuf} \
    --destdir $destdir --bert-model-name $bert_name --workers $w
fi
echo

