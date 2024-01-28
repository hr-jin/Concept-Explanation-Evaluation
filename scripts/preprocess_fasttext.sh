#!/bin/bash
while IFS= read -r LINE; do
  LINEVARS=( $LINE )
  CORPUS=${LINEVARS[0]}
  EMBEDDING=cc.${LINEVARS[1]}.300.bin
  CMD="python ./ud_treebank_utils/preprocess_treebank.py $CORPUS --skip-existing --fasttext $EMBEDDING"
  echo "$CMD"
  eval $CMD
done < "./languages_common_selected.lst"
