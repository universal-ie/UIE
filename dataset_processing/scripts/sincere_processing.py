#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os


def make_new_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def processing_sentence(sentence):
    new_sentence = {'tokens': sentence['tokens'], 'span_pair_list': sentence['relations']}
    span_list = list()
    for entity in sentence['entities']:
        span = entity
        span['end'] -= 1
        span_list += [span]
    new_sentence['span_list'] = span_list
    return new_sentence


def main():
    raw_folder = "data/sincere"
    new_folder = "data/relation"
    for dataset_name in ['ace05', 'conll04']:
        dataset_file_name = os.path.join(raw_folder, dataset_name) + '.json'

        output_folder = os.path.join(new_folder, dataset_name)
        make_new_folder(output_folder)

        dataset = json.load(open(dataset_file_name))
        for split_name in dataset:
            split_filename = os.path.join(output_folder, split_name + '.jsonlines')

            with open(split_filename, 'w') as output:
                for sentence in dataset[split_name]:
                    output.write(json.dumps(processing_sentence(sentence)) + '\n')


if __name__ == "__main__":
    main()
