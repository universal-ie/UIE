#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
import sys


def walk_folder(folder_name, end_str='.json'):
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            file_name = os.path.join(root, file)
            if file_name.endswith(end_str):
                yield file_name


def main(annotation_folder, content_folder):
    if not os.path.exists(content_folder):
        os.makedirs(content_folder, exist_ok=True)

    for annotation_filename in walk_folder(annotation_folder):
        content_filename = annotation_filename.replace(annotation_folder, content_folder).replace('.json', '.text')
        text = open(annotation_filename).read()

        if text == "":
            print("Empty file %s" % annotation_filename)
            continue

        try:
            content = json.load(open(annotation_filename))['content']
        except:
            print(annotation_filename)
            continue

        with open(content_filename, 'w') as output:
            output.write(content)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
