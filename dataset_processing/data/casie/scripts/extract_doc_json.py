#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
import sys
from collections import defaultdict, Counter
from pprint import pprint

from tqdm import tqdm


def fix_offset(annotation, span):
    if annotation['content'][span['startOffset'] - 1:span['endOffset']] == span['text']:
        span['startOffset'] -= 1

    if annotation['content'][span['startOffset']:span['endOffset'] + 1] == span['text']:
        span['endOffset'] += 1

    if annotation['content'][span['startOffset'] - 1:span['endOffset'] + 1] == span['text']:
        span['endOffset'] += 1
        span['startOffset'] -= 1

    if annotation['content'][span['startOffset'] + 1:span['endOffset'] - 1] == span['text']:
        span['endOffset'] -= 1
        span['startOffset'] += 1

    if annotation['content'][span['startOffset'] - 1:span['endOffset'] - 1] == span['text']:
        span['endOffset'] -= 1
        span['startOffset'] -= 1

    if annotation['content'][span['startOffset'] - 2:span['endOffset'] - 2] == span['text']:
        span['endOffset'] -= 2
        span['startOffset'] -= 2

    if annotation['content'][span['startOffset']:span['endOffset']] != span['text']:
        print(annotation['sourcefile'], span['startOffset'], span['endOffset'])
        print(annotation['content'][span['startOffset']:span['endOffset']] + '||' + span['text'])

    return span


class CharSeq:
    def __init__(self, start, end, text, docid=None, init_use_end=False):
        assert type(start) == int
        assert type(end) == int
        self.start = start
        # ACE End is End Char Offset, use_end is use in python
        # text = doc[start:use_end]
        if init_use_end:
            self.end = end - 1
            self.use_end = end
        else:
            self.end = end
            self.use_end = end + 1

        self.length = self.end - self.start + 1
        self.docid = docid
        self.text = text

    def __str__(self):
        return "(%s)[%s-%s-%s]" % (self.text, self.start, self.end, self.docid)

    def to_json(self):
        return {'offset': self.start,
                'length': self.length,
                'text': self.text}

    @staticmethod
    def from_stanford_token(token, docid=None):
        return CharSeq(start=token['characterOffsetBegin'],
                       end=token['characterOffsetEnd'],
                       text=token['originalText'],
                       docid=docid,
                       init_use_end=True)

    def exact_match(self, other, check_doc=True):
        """
        :param other: CharSeq
        :param check_doc: Boolean
        :return:
        """
        if check_doc:
            assert self.docid == other.docid
        if self.start == other.start and self.end == other.end:
            assert self.text == other.text
            return True
        return False

    def partial_match(self, other, check_doc=True):
        """
        :param other: CharSeq
        :param check_doc: Boolean
        :return:
        """
        if check_doc:
            assert self.docid == other.docid
        if self.start < other.start and self.end < other.start:
            return False
        elif other.start < self.start and other.end < self.start:
            return False
        else:
            return True


def get_token_charseq_dict(sentences, doc_id):
    token_charseq_dict = dict()

    for senid, sentence in enumerate(sentences):
        for tokenid, token in enumerate(sentence['tokens']):
            charseq = CharSeq.from_stanford_token(token, docid=doc_id)
            token_charseq_dict[(senid, tokenid)] = charseq

    return token_charseq_dict


def find_span_tokens(token_charseq_dict, span, exact_match=False, verbose=True):
    tokens = list()
    for token_iden, charseq in token_charseq_dict.items():
        if exact_match:
            if charseq.exact_match(span):
                tokens.append(token_iden)
        else:
            if charseq.partial_match(span):
                tokens.append(token_iden)
    tokens.sort()

    if len(tokens) == 0 and verbose:
        sys.stderr.write("Miss span: [%s][%s-%s-%s]\n" % (span.text, span.docid, span.start, span.end))
    return tokens


def token_to_json(token):
    if len(token['originalText']) != (token['characterOffsetEnd'] - token['characterOffsetBegin']):
        print('Length is not Match', token)
        # exit(1)
    return {'characterOffsetBegin': token['characterOffsetBegin'],
            'characterOffsetEnd': token['characterOffsetEnd'],
            'word': '_'.join(token['word'].split()),
            'originalText': token['originalText'],
            # 'lemma': token['lemma'],
            # 'pos': token['pos'],
            # 'ner': token['ner'],
            }


def process_sentence(sentence):
    tokens = [token_to_json(token) for token in sentence['tokens']]
    # parsed_tree = sentence.get('parse', None)
    # depparsed_tree = sentence.get('basicDependencies', None)
    return {'tokens': tokens,
            # 'parse': parsed_tree,
            # 'depparse': depparsed_tree,
            'span': (tokens[0]['characterOffsetBegin'],
                     tokens[-1]['characterOffsetEnd'])
            }


def span_to_dict(annotation, span, token_charseq_dict):
    span_start, span_end = span['startOffset'], span['endOffset']
    return {
        'text': annotation['content'][span_start:span_end],
        'span': (span_start, span_end - 1),
        'tokens': find_span_tokens(token_charseq_dict,
                                   span=CharSeq(start=span_start,
                                                end=span_end - 1,
                                                text=annotation['content'][span_start:span_end],
                                                docid=annotation['sourcefile']))

    }


def main():
    annotation_folder = "raw_data/annotation"
    content_folder = "raw_data/content"
    corenlp_folder = "raw_data/corenlp"
    output_filename = "raw_data/casie.jsonlines"

    type_dict = defaultdict(Counter)

    doc_list = list()

    for doc_id in tqdm(range(20000)):
        annotation_filename = f"{annotation_folder}/{doc_id}.json"
        content_filename = f"{content_folder}/{doc_id}.text"
        corenlp_filename = f"{corenlp_folder}/{doc_id}.text.json"

        if not os.path.exists(annotation_filename):
            continue

        content = open(content_filename).read()
        annotation = json.load(open(annotation_filename))
        corenlp = json.load(open(corenlp_filename))

        sentences = [process_sentence(sentence) for sentence in corenlp['sentences']]
        token_charseq_dict = get_token_charseq_dict(sentences=sentences, doc_id=annotation['sourcefile'])

        assert len(annotation['cyberevent']) == 1

        hopper_list = list()

        for hopper_id, hopper in enumerate(annotation['cyberevent']['hopper']):

            hopper_dict = {
                'id': "%s-%s" % (doc_id, hopper_id),
                'mentions': list()
            }

            type_dict['relation'].update([hopper.get('relation', None)])
            for event_id, event in enumerate(hopper['events']):

                type_dict['type'].update([event['type']])
                type_dict['subtype'].update([event['subtype']])
                type_dict['realis'].update([event['realis']])

                nugget = event['nugget']
                if content[nugget['startOffset']:nugget['endOffset']] != nugget['text']:
                    nugget = fix_offset(annotation, nugget)

                event_dict = {
                    'id': "%s-%s-%s" % (doc_id, hopper_id, event_id),
                    'type': event['type'],
                    'subtype': event['subtype'],
                    'realis': event['realis'],
                    'nugget': span_to_dict(annotation, span=nugget, token_charseq_dict=token_charseq_dict),
                    'arguments': list(),
                }

                for argument_id, argument in enumerate(event.get('argument', [])):

                    type_dict['role'].update([argument['role']['type']])
                    type_dict['argument_type'].update([argument['type']])

                    if content[argument['startOffset']:argument['endOffset']] != argument['text']:
                        argument = fix_offset(annotation, argument)

                    argument_dict = {
                        'id': "%s-%s-%s-%s" % (doc_id, hopper_id, event_id, argument_id),
                        'role': argument['role']['type'],
                        'filler_type': argument['type']
                    }

                    argument_dict.update(span_to_dict(annotation, span=argument, token_charseq_dict=token_charseq_dict))
                    event_dict['arguments'] += [argument_dict]

                hopper_dict['mentions'] += [event_dict]
            hopper_list += [hopper_dict]
        doc_list += [{'id': annotation['sourcefile'],
                      'sentences': sentences,
                      'text': content,
                      'stanford_coref': corenlp.get('corefs', {}),
                      'event': hopper_list,
                      'info': annotation['info']
                      }]

    for name, value in type_dict.items():
        print(name, sum(value.values()))
        pprint(value)

    with open(output_filename, 'w') as output:
        for doc in doc_list:
            output.write(json.dumps(doc, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
