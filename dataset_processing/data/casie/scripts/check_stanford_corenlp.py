#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os


def walk_folder(folder_name, end_str='.json'):
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            file_name = os.path.join(root, file)
            if file_name.endswith(end_str):
                yield file_name


def main(source_folder, corenlp_folder):
    safe = True
    mismatch_set = set()
    for text_file in walk_folder(source_folder, '.txt'):
        json_file = text_file.replace(source_folder, corenlp_folder) + '.json'

        if not os.path.exists(json_file):
            print("Not found:", json_file)
            safe = False
            continue

        text = open(text_file).read()
        document = json.load(open(json_file))

        for sentence in document['sentences']:
            for token in sentence['tokens']:
                token_start = token['characterOffsetBegin']
                token_end = token['characterOffsetEnd']
                if token['originalText'] != text[token_start:token_end].replace('\n', ''):
                    safe = False
                    mismatch_set.add(text_file)

    for filename in mismatch_set:
        print('mismatch doc:', filename)
    
    if safe:
        print('All Clear')


if __name__ == "__main__":
    main('raw_data/content', 'raw_data/corenlp')
