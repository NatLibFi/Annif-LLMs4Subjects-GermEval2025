import sys
import gzip
import zipfile
import json
import collections
from itertools import islice
import annif_client

# read command line parameters
port = sys.argv[1]     # TCP port where Annif REST API is available
project = sys.argv[2]  # Annif project ID, where LANG stands for language ID
langs = sys.argv[3]    # can be "de", "en" or "de,en"
outfile = sys.argv[4]  # zip file where predictions are written as JSON files
infiles = sys.argv[5:] # gzipped JSONL file(s) with input documents

api_base = f"http://127.0.0.1:{port}/v1/"
BATCH_SIZE = 32
LIMIT = 50

annif = annif_client.AnnifClient(api_base=api_base)

def read_jsonl_files(filenames):
    for fn in filenames:
        with gzip.open(fn, "rt") as inf:
            for line in inf:
                rec = json.loads(line)
                yield rec

def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

def shorten(uri):
    return uri.replace('https://d-nb.info/gnd/', 'gnd:')


def get_suggestions(batch):
    doc_counters = collections.defaultdict(collections.Counter)
    for lang in langs.split(','):
        project_id = project.replace('LANG', lang)
        documents = [{'document_id': rec['filename'], 'text': rec[f"text_{lang}"]} for rec in batch]
        response = annif.suggest_batch(project_id, documents, limit=LIMIT)
        for doc_results in response:
            for result in doc_results["results"]:
                doc_counters[doc_results['document_id']][shorten(result['uri'])] += result['score']
    return {docid: [c[0] for c in counter.most_common(LIMIT)] for docid, counter in doc_counters.items()}


with zipfile.ZipFile(outfile, 'w') as zip:
    for batch in batched(read_jsonl_files(infiles), BATCH_SIZE):
        batch_suggestions = get_suggestions(batch)
        for filename, suggestions in batch_suggestions.items():
            fn = filename.replace('.jsonld', '.json')
            data = json.dumps({"dcterms:subject": suggestions}, indent="\t")
            zip.writestr(fn, data=data)

        # uncomment line below to test on only one batch
        # break
