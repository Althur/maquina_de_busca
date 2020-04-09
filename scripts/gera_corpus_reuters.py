 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:04:46 2020

@author: arthurrizzo
"""

import json
from nltk.corpus import reuters
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('dest_file',
                        help='Arquivo onde armazenar o arquivo do corpus.')
    args = parser.parse_args()

    docs = {}
    for fileid in reuters.fileids():
        docs[fileid] = reuters.raw(fileid)

    with open(args.dest_file, 'w') as file:
        json.dump(docs, file, indent=4)


if __name__ == '__main__':
    main()