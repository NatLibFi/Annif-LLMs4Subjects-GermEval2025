import collections
import json
import zipfile
import os
import os.path
import sys

# read command line parameters
de_pred = sys.argv[1]  # input JSONL file with German language predictions
en_pred = sys.argv[2]  # input JSONL file with English language predictions
outbase = sys.argv[3]  # output file basename

LIMIT = 20  # how many predictions to select per document

# possible weights for the German prediction (English weight is 1-x)
DE_WEIGHTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

def read_predictions(fn):
    preds = {}
    with open(fn) as infile:
        for line in infile:
            pred = json.loads(line)
            preds[pred['file']] = pred['suggestions']
    return preds

# transform path, e.g. "dev/Book/en/12345.jsonld" -> "subtask2/Book/en/12345.json"
def transform_path(filepath):
    parts = filepath.replace('.jsonld', '.json').split(os.sep)  # change extension
    stripped_path = os.path.join(*parts[1:])  # Remove the first part
    return os.path.join("subtask2", stripped_path)  # prepend "subtask2/"

# merge predictions for a file using the given weight
def merge_predictions(filename, de_weight):
    counter = collections.Counter()
    for preds, weight in zip([de_preds, en_preds], [de_weight, 1-de_weight]):
        for concept, score in preds.get(filename, {}).items():
            counter[concept] += score * weight
    return [c[0] for c in counter.most_common(LIMIT)]

# read the predictions
de_preds = read_predictions(de_pred)
en_preds = read_predictions(en_pred)

# consolidate filenames into a single set
filenames = set()
filenames.update(de_preds.keys())
filenames.update(en_preds.keys())

# produce output zipfiles
for de_weight in DE_WEIGHTS:
    outfile = f"{outbase}-{de_weight}.zip"
    with zipfile.ZipFile(outfile, 'w') as outzip:
        for filename in filenames:
            fn = transform_path(filename)
            subjects = merge_predictions(filename, de_weight)
            data = json.dumps({"dcterms:subject": subjects}, indent="\t")
            outzip.writestr(fn, data=data)
