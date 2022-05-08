#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import json


def main():
    data_file = "raw_data/casie.jsonlines"
    output_folder = "data"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    instance_list = [json.loads(line) for line in open(data_file)]

    instance_list = [(instance['info']['date'], instance['id'], instance) for instance in instance_list]
    instance_list.sort(reverse=True)

    data_split = {
        'test': (0, 200),
        'dev': (200, 300),
        'train': (300, len(instance_list)),
    }

    for split_name, (start, end) in data_split.items():
        with open(os.path.join(output_folder, '%s.jsonlines' % split_name), 'w') as output:
            for instance in instance_list[start: end]:
                output.write(json.dumps(instance[2]) + '\n')


if __name__ == "__main__":
    main()
