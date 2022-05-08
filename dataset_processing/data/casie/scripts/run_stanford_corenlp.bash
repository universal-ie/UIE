#!/usr/bin/env bash
# -*- coding:utf-8 -*-

export MEMORY_LIMIT=64g
export STANFORD_CORENLP_PATH=./corenlp/stanford-corenlp-4.1.0
export STANFORD_CORENLP_MODEL_PATH=./corenlp/models
export THREAD_NUM=32


java -mx${MEMORY_LIMIT} \
  -cp "${STANFORD_CORENLP_PATH}/*:${STANFORD_CORENLP_PATH}/lib/*:${STANFORD_CORENLP_PATH}/liblocal/*:${STANFORD_CORENLP_MODEL_PATH}/*" \
  edu.stanford.nlp.pipeline.StanfordCoreNLP \
  -annotators tokenize,cleanxml,ssplit \
  -outputFormat json \
  -threads ${THREAD_NUM} \
  -outputDirectory raw_data/corenlp \
  -file raw_data/content


python scripts/check_stanford_corenlp.py
