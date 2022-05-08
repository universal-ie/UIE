#!/usr/bin/env bash
# -*- coding:utf-8 -*-

python scripts/generate_content.py raw_data/annotation raw_data/content
bash scripts/run_stanford_corenlp.bash
python scripts/extract_doc_json.py
python scripts/split_data.py
tree data/