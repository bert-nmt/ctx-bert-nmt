#!/usr/bin/env bash
# Paras: lng, startofdoc
# Execute this script in $prep, e.g. iwslt14.tokenized.de-en

lng=$1
startofdoc=${2:-"startofdocumentplaceholder"}

echo "src lng $lng"

for sub  in train valid test
do
    inp=tmp/${sub}.${lng}.with_startofdoc
    oup=${sub}.bert.${lng}
    sent=${sub}.bert.sent.${lng}
    echo "$inp  ==>  $oup  ==>  $sent"
    # sed -r 's/(@@ )|(@@ ?$)//g' ${sub}.${lng} > ${sub}.bert.${lng}.tok
    ../mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng < $inp > $oup
    # rm ${sub}.bert.${lng}.tok
    cat $oup | grep -v ^$startofdoc > $sent
done
