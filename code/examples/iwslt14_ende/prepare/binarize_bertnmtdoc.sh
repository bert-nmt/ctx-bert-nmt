# paras: ctx, path, src, tgt, workers
ctx=$1
path=$2
TEXT=$path/iwslt14.tokenized.de-en
src=${3:-de}
tgt=${4:-en}
w=${5:-10}

for subset in train valid test; do
  for lng in $src $tgt; do
    echo "$subset.$lng => ${subset}${ctx}.${lng}"
    cp $TEXT/$subset.$lng $TEXT/${subset}${ctx}.${lng}
  done
done

export PYTHONPATH=$FAIRSEQ:$PYTHONPATH

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

if [ "$src" = "de" ]; then
  bert_name=bert-base-german-cased
elif [ "$src" = "en" ]; then
  bert_name=bert-base-uncased
else
  echo "Invalid src language $src, no corresponding bert model"
  exit
fi
echo "Using BERT $bert_name"

if [ -d $destdir ] && [ ! -z "$(ls -A $destdir)" ]; then
  echo "$destdir exists and non-empty"
  exit
fi
echo "bertnmtdoc: dest dir $destdir"
mkdir $destdir -p
python $FAIRSEQ/preprocess.py --source-lang $src --target-lang $tgt --bert-input ${bert_input} \
  --joined-dictionary \
  --trainpref $TEXT/train${ctx} --validpref $TEXT/valid${ctx} --testpref $TEXT/test${ctx} \
  --destdir $destdir --bert-model-name $bert_name --workers $w

