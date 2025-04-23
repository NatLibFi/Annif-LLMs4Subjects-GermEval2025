import glob
import json
import sys
import time
import os.path
import zipfile
from itertools import islice
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

LLM_SYSTEM_PROMPT = "You are a professional translator specialized in translating bibliographic metadata."
LLM_PROMPT = """
Your task is to ensure that the given document title and description are in <LANGUAGE> language, translating the text if necessary.
If the text is already in <LANGUAGE>, do not change or summarize it, keep it all as it is.

Respond with only the text, nothing else.

Give this title and description in <LANGUAGE>:
""".strip()

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 4096
TEMPERATURE = 0.3
REPETITION_PENALTY = 1.1
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.95
MAX_BATCHED_TOKENS = 16384

# Read input from zip file and write output to JSONL file alongside the zip
source_filename = sys.argv[1]
lang = sys.argv[2]
dest_filename = source_filename.replace('.zip', f"-{lang}.jsonl")

# Initialize vLLM engine and tokenizer
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=GPU_MEM_UTIL,
    max_model_len=MAX_MODEL_LEN,
    max_num_batched_tokens=MAX_BATCHED_TOKENS,
    enable_prefix_caching=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def generate_messages(record, language):
    prompt = LLM_PROMPT.replace('<LANGUAGE>', language) + "\n\n" + \
        record['title'] + "\n\n" + record['desc']

    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    return messages


def messages_to_token_ids(messages):
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


# Function to process a batch of records
def process_batch(batch):
    prompt_token_ids = []
    languages = []

    for record in batch:
        messages = generate_messages(record, 'German')
        prompt_token_ids.append(messages_to_token_ids(messages))
        languages.append('de')
        messages = generate_messages(record, 'English')
        prompt_token_ids.append(messages_to_token_ids(messages))
        languages.append('en')

    new_records = {'en': [], 'de': []}

    sampling_params=SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, repetition_penalty=REPETITION_PENALTY)
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

    generated_text = [output.outputs[0].text for output in outputs]
    for lang, text in zip(languages, generated_text):
        new_records[lang].append(text)

    return new_records['en'], new_records['de']

def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

def read_zip(filename, lang):
    with zipfile.ZipFile(filename) as zf:
        for fn in zf.namelist():
            if f"/{lang}/" not in fn or not fn.endswith('.jsonld'):
                continue
            with zf.open(fn) as jsonld_file:
                yield read_jsonld_file(jsonld_file, fn)

def read_jsonld_file(jsonld_file, filename):
    data = json.load(jsonld_file)["@graph"]
    title = abstract = uris = None
    for field in data:
        if "title" in field.keys():
            title = clean_and_combine(field["title"])
        if "abstract" in field.keys():
            abstract = clean_and_combine(field["abstract"])
        if "dcterms:subject" in field.keys():
            if isinstance(field["dcterms:subject"], list):
                uris = [subj["@id"] for subj in field["dcterms:subject"]]
            else:
                uris = [field["dcterms:subject"]["@id"]]
    rec = {"filename": filename, "title": title, "desc": abstract}
    if uris:
        rec["subjects"] = [uri.replace('gnd:', "https://d-nb.info/gnd/") for uri in uris]
    return rec

def clean_and_combine(input):
    if isinstance(input, list):
        return " ¤ ".join([" ".join(i.split()) for i in input])
    return input

# Process input lines in batches
batch_size = 256

starttime = time.time()
ndocs = 0

with open(dest_filename, 'w') as dest_file:
    for batch in batched(read_zip(source_filename, lang), batch_size):
        ndocs += len(batch)
        en_records, de_records = process_batch(batch)

        for orig_rec, en_rec, de_rec in zip(batch, en_records, de_records):
            rec = {}
            rec['filename'] = orig_rec['filename']
            rec['subjects'] = orig_rec['subjects']
            rec['title'] = orig_rec['title']
            rec['desc'] = orig_rec['desc']
            rec['text_en'] = en_rec
            rec['text_de'] = de_rec
            json.dump(rec, dest_file)
            dest_file.write('\n')

        dest_file.flush()

elapsed = time.time() - starttime
print(f"Time taken: {elapsed} seconds ({elapsed/ndocs} seconds per document), batch size {batch_size}")
