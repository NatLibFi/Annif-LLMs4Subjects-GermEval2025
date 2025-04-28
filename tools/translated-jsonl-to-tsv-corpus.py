#!/usr/bin/env python

import sys
import json
import gzip
import os.path

if len(sys.argv) < 3:
    print(f"usage: {sys.argv[0]} <output-directory> <infile> [<infile2> ...]", file=sys.stderr)
    print("input files are expected to be in jsonl.gz format", file=sys.stderr)
    sys.exit(1)

outdir = sys.argv[1]
infiles = sys.argv[2:]

# determine output file paths within output directory
orig_fn = os.path.join(outdir, "orig.tsv.gz")
en_fn = os.path.join(outdir, "en.tsv.gz")
de_fn = os.path.join(outdir, "de.tsv.gz")

def clean_text(text):
    return ' '.join(text.strip().split())


def write_tsv(f, text, uris):
    uris_fmt = ' '.join([f"<{uri}>" for uri in uris])
    print(f"{text}\t{uris_fmt}", file=f)


def process_file(in_f, orig_f, en_f, de_f):
    for line in in_f:
        rec = json.loads(line)
        orig_text = f"{rec['title']} ¤ {rec['desc']}"
        en_text = clean_text(rec['text_en'].replace('\n\n', ' ¤ '))
        de_text = clean_text(rec['text_de'].replace('\n\n', ' ¤ '))
        write_tsv(orig_f, orig_text, rec['subjects'])
        write_tsv(en_f, en_text, rec['subjects'])
        write_tsv(de_f, de_text, rec['subjects'])


with gzip.open(orig_fn, "wt") as orig_f, \
     gzip.open(en_fn, "wt") as en_f, \
     gzip.open(de_fn, "wt") as de_f:
    for in_fn in infiles:
        with gzip.open(in_fn) as in_f:
            process_file(in_f, orig_f, en_f, de_f)
