#!/usr/bin/env bash
# Paras: lng, startofdoc
# Execute this script in $prep, e.g. iwslt14.tokenized.de-en

lng=$1
final=${2:-"final_1wBPE_fix"}
startofdoc=${3:-"startofdocumentplaceholder"}

echo "src lng $lng"

for sub  in train valid test
do
    inp=tmp/${sub}.${lng}.with_startofdoc
    oup=$final/${sub}.bert.${lng}
    sent=$final/${sub}.bert.sent.${lng}
    echo "$inp  ==>  $oup  ==>  $sent"
    # sed -r 's/(@@ )|(@@ ?$)//g' ${sub}.${lng} > ${sub}.bert.${lng}.tok
    if [[ $lng != zh* ]]; then
        perl mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng < $inp > $oup
    else
        cat $inp | sed "s/ //g" > $oup
    fi
    # rm ${sub}.bert.${lng}.tok
    cat $oup | grep -v ^$startofdoc > $sent
done
