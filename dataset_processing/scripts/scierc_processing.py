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
    raw_folder = "data/dygiepp/scierc"
    new_folder = "data/relation/scierc"
    for split in ['train', 'dev', 'test']:
        raw_file = os.path.join(raw_folder, f"{split}.json")
        new_file = os.path.join(new_folder, f"{split}.jsonlines")
        print(f"convert {raw_file} to {new_file}")
        with open(new_file, 'w') as output:
            for line in open(raw_file):
                instance = json.loads(line)
                sentence_start = 0
                for sentence, ner, relation in zip(instance['sentences'],
                                                   instance['ner'],
                                                   instance['relations']):
                    ner = [[x[0] - sentence_start, x[1] - sentence_start, x[2]]
                           for x in ner]
                    relation = [[x[0] - sentence_start, x[1] - sentence_start,
                                x[2] - sentence_start, x[3] - sentence_start, x[4]]
                                for x in relation]

                    span_list = [{'start': x[0], 'end': x[1], 'type': x[2]}
                                 for x in ner]

                    ner_dict = {(x['start'], x['end']): i
                                for i, x in enumerate(span_list)}

                    span_pair_list = list()
                    for rel in relation:
                        head_index = ner_dict[(rel[0], rel[1])]
                        tail_index = ner_dict[(rel[2], rel[3])]
                        span_pair_list += [{
                            'head': head_index,
                            'tail': tail_index,
                            'type': rel[4],
                        }]

                    sentence_start += len(sentence)
                    spannet_instance = {
                        'tokens': sentence,
                        'span_list': span_list,
                        'span_pair_list': span_pair_list
                    }
                    output.write(
                        json.dumps(spannet_instance, ensure_ascii=False) + '\n'
                    )


if __name__ == "__main__":
    main()
