#!/usr/bin/env bash
# -*- coding:utf-8 -*-

mkdir corenlp

# Length: 504773765 (481M) [application/zip]
wget -P corenlp/backup http://nlp.stanford.edu/software/stanford-corenlp-4.1.0.zip

unzip -d corenlp corenlp/backup/stanford-corenlp-4.1.0.zip

# Length: 670717962 (640M) [application/java-archive]
wget -P corenlp/models http://nlp.stanford.edu/software/stanford-corenlp-4.1.0-models-english.jar
