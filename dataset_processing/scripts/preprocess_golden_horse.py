#!/usr/bin/env python
# -*- coding:utf-8 -*-


def convert_file(filename, output_filename, ignore_class=None):
    with open(output_filename, 'w') as output:
        for line in open(filename):
            att = line.strip().split('\t')
            if len(att) < 2:
                output.write('\n')
            else:
                token, tag = att
                token = token[0]
                if ignore_class is not None:
                    if tag.endswith(ignore_class):
                        tag = 'O'
                output.write(f'{token}\t{tag}\n')


def main():
    convert_file("data/golden-horse/data/weiboNER_2nd_conll.train", "data/golden-horse/data/weiboNER_2nd_conll.train.word_tag",)
    convert_file("data/golden-horse/data/weiboNER_2nd_conll.dev", "data/golden-horse/data/weiboNER_2nd_conll.dev.word_tag",)
    convert_file("data/golden-horse/data/weiboNER_2nd_conll.test", "data/golden-horse/data/weiboNER_2nd_conll.test.word_tag",)

    convert_file("data/golden-horse/data/weiboNER_2nd_conll.train", "data/golden-horse/data/weiboNER_2nd_conll.train.word_tag.nom", ignore_class='NAM')
    convert_file("data/golden-horse/data/weiboNER_2nd_conll.dev", "data/golden-horse/data/weiboNER_2nd_conll.dev.word_tag.nom", ignore_class='NAM')
    convert_file("data/golden-horse/data/weiboNER_2nd_conll.test", "data/golden-horse/data/weiboNER_2nd_conll.test.word_tag.nom", ignore_class='NAM')

    convert_file("data/golden-horse/data/weiboNER_2nd_conll.train", "data/golden-horse/data/weiboNER_2nd_conll.train.word_tag.nam", ignore_class='NOM')
    convert_file("data/golden-horse/data/weiboNER_2nd_conll.dev", "data/golden-horse/data/weiboNER_2nd_conll.dev.word_tag.nam", ignore_class='NOM')
    convert_file("data/golden-horse/data/weiboNER_2nd_conll.test", "data/golden-horse/data/weiboNER_2nd_conll.test.word_tag.nam", ignore_class='NOM')


if __name__ == "__main__":
    main()
