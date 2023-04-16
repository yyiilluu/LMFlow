"""
python src/lmflow/data_gen/generate_data_using_openai.py \
    --collection /Users/yilu/workspace/repos/LMFlow/data/wa/wa_glossary_clean_merged.txt \
    --output_dir /Users/yilu/workspace/repos/LMFlow/data/wa/glossary
"""
import argparse
import json
import os

import openai
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', type=str)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--engine', type=str, default='curie')
    parser.add_argument('--max_examples', type=int, default=100000,
                        help='Maximum number of documents to read from the collection.')
    parser.add_argument('--max_tokens', type=int, default=64, help='Max tokens to be generated.')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature. Zero means greedy decoding.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--min_doc_chars', type=int, default=0,
                        help='Minimum number of chars an input document must have.')
    parser.add_argument('--max_doc_chars', type=int, default=100000,
                        help='Maximum number of chars an input document must have.')
    parser.add_argument('--sleep_time', type=float, default=1.5,
                        help='Time to wait between API calls, in seconds.')
    parser.add_argument('--good_bad', action='store_true',
                        help='The model should produce a good question followed by a bad question.')
    parser.add_argument('--include_doc_probs', action='store_true',
                        help='Wheter or not to save the tokens probabilities produeced by the model.')

    args = parser.parse_args()

    openai.api_key = ""
    num_examples_so_far = 0
    skip_doc_ids = set()

    n_docs_skipped = 0
    os.makedirs(args.output_dir, exist_ok=True)
    error_counter = 0
    with open(args.collection) as f:
        with open(os.path.join(args.output_dir, "generated_qa.jsonl"), 'w') as fout, \
            open(os.path.join(args.output_dir, "failed_doc_qa.jsonl"), 'w') as ferr:
            lines = [l for l in f]
            progress_bar = tqdm(total=len(lines))
            progress_bar.n = num_examples_so_far

            if len(lines) >= args.max_examples:
                print(f"More than max examples. It has {len(lines)} but max is "
                      f"{args.max_examples}")
                lines = lines[:args.max_examples]

            for line_num, line in enumerate(lines):

                if num_examples_so_far >= args.max_examples:
                    break

                doc_text = line.strip()

                if len(doc_text) < args.min_doc_chars:
                    n_docs_skipped += 1
                    print(f'Skipping due to min len. Skipped {n_docs_skipped} docs so far')
                    continue

                if len(doc_text) > args.max_doc_chars:
                    n_docs_skipped += 1
                    print(f'Skipping due to max len. Skipped {n_docs_skipped} docs so far')
                    continue
                output = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a real estate agent who wants to "
                                                      "predict what kinds of question your "
                                                      "client would ask"},
                        {"role": "user", "content": f"what are the questions that might be asked "
                                                    """by real estate client and corresponding answers, which can be 
                                                    found by the following document and  in 
                                                    json format like [{"question": "", "answer": ""}, {"question": "", "answer": ""}].
""" + f" Document: {doc_text}"},
                    ]
                )

                message = output['choices'][0]['message']['content']
                try:
                    message_json = json.loads(message)
                except:
                    json.dump({
                        "doc": doc_text,
                        "output": message
                    }, ferr)
                    print(f"Failed to generated proper json, counter {error_counter}")
                    continue

                index_start = 0
                index_end = 0

                output_dict = {
                    'doc_text': doc_text,
                    'qas': message_json,
                }

                fout.write(json.dumps(output_dict) + '\n')
                fout.flush()
                if line_num & (line_num - 1) == 0 or line_num % 1000 == 0:
                    # LOG every power of 2 or 1000 steps.
                    print(f'Document: {doc_text}\nQAS: {message_json}\n')

                num_examples_so_far += 1
                progress_bar.update(1)

    print('Done!')
