#!/usr/bin/env bash
# -*- coding:utf-8 -*-

git clone git@github.com:Ebiquity/CASIE.git || exit 1

cp -r CASIE/data raw_data

# ignore file with some error
for file_id in 999 10001 10002
do
  rm raw_data/source/${file_id}.txt raw_data/annotation/${file_id}.json
done
