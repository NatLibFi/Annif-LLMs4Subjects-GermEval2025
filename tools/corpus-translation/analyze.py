import sys
import re
import csv
from collections import defaultdict

def analyze_input(input_text):
    pattern = re.compile(r'Translating corpus file: .* .* language (\w{2}) using model (\w+)\nTime taken: ([\d.]+) seconds for (\d+) documents')
    matches = pattern.findall(input_text)

    results = defaultdict(lambda: {'total_time': 0, 'total_documents': 0})
    for match in matches:
        language, model_id, time_taken, num_documents = match
        key = (model_id, language)
        results[key]['total_time'] += float(time_taken)
        results[key]['total_documents'] += int(num_documents)

    return results

if __name__ == "__main__":
    input_text = sys.stdin.read()
    analysis_results = analyze_input(input_text)

    writer = csv.writer(sys.stdout)

    for (model_id, language), data in analysis_results.items():
        writer.writerow([language, model_id, data['total_documents'] / data['total_time']])

