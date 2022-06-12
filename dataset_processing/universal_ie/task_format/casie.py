#!/usr/bin/env python
# -*- coding:utf-8 -*-

from collections import defaultdict, Counter
import json
from typing import List
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import tokens_to_str
from universal_ie.ie_format import Entity, Event, Label, Sentence, Span


class CASIE(TaskFormat):
    def __init__(self, sentence_dict, language="en"):
        super().__init__(language=language)
        self.sent_id = sentence_dict["sent_id"]
        self.tokens = sentence_dict["tokens"]
        self.entities = sentence_dict["entity_mentions"]
        self.events = sentence_dict["event_mentions"]

    def generate_instance(self):
        entities = {}
        events = {}

        for entity in self.entities:
            indexes = entity["indexes"]
            tokens = [self.tokens[id] for id in indexes]
            entities[entity["id"]] = Entity(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id,
                ),
                label=Label(entity["type"]),
                text_id=self.sent_id,
                record_id=entity["id"],
            )

        for event in self.events:
            indexes = event["trigger"]["indexes"]
            tokens = [self.tokens[id] for id in indexes]
            events[event["id"]] = Event(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id,
                ),
                label=Label(event["type"]),
                args=[
                    (Label(x["role"]), entities[x["id"]])
                    for x in event["arguments"]
                ],
                text_id=self.sent_id,
                record_id=event["id"],
            )

        return Sentence(
            tokens=self.tokens,
            entities=entities.values(),
            events=events.values(),
            text_id=self.sent_id,
        )

    @staticmethod
    def load_from_file(filename, language="en") -> List[Sentence]:
        sentence_list = []
        counter = Counter()

        with open(filename) as fin:
            for line in fin:
                doc = json.loads(line.strip())

                entity_mentions = defaultdict(list)
                event_mentions = defaultdict(list)

                for event in doc["event"]:
                    for mention in event["mentions"]:
                        nugget = mention["nugget"]
                        sent_id = nugget["tokens"][0][0]

                        if nugget["tokens"][0][0] != nugget["tokens"][-1][0]:
                            counter.update(['cross_sentence_trigger'])
                            continue

                        event_mention = {
                            "id": mention["id"],
                            "type": mention["subtype"],
                            "trigger": {"indexes": [x[1] for x in nugget["tokens"]],},
                            "arguments": [],
                        }
                        counter.update(['event mention'])

                        for argument in mention["arguments"]:
                            if argument["tokens"][0][0] != argument["tokens"][-1][0]:
                                counter.update(['cross_sentence_arg'])
                                continue
                            arg_sent_id = argument["tokens"][0][0]
                            entity_mention = {
                                "id": argument["id"],
                                "indexes": [x[1] for x in argument["tokens"]],
                                "type": argument["filler_type"],
                            }
                            entity_mentions[arg_sent_id].append(entity_mention)
                            counter.update(['entity'])
                            if arg_sent_id == sent_id:
                                event_mention["arguments"].append(
                                    {
                                        "id": argument["id"],
                                        "trigger": {
                                            "indexes": [x[1] for x in nugget["tokens"]],
                                        },
                                        "role": argument["role"],
                                    }
                                )
                                counter.update(['argument'])
                            else:
                                counter.update(['cross_sentence_tri_arg'])

                        event_mentions[sent_id].append(event_mention)

                for sent_id, sentence in enumerate(doc["sentences"]):
                    tokens = [token["word"] for token in sentence["tokens"]]

                    sentence_dict = {
                        "sent_id": sent_id,
                        "tokens": tokens,
                        "entity_mentions": entity_mentions[sent_id],
                        "event_mentions": event_mentions[sent_id],
                    }
                    instance = CASIE(
                        sentence_dict, language=language
                    ).generate_instance()

                    sentence_list.append(instance)
                    counter.update(['sentence'])

        print(filename, counter)
        return sentence_list
