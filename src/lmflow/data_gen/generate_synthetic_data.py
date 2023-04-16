"""
python ./src/lmflow/data_gen/generate_synthetic_data.py \
    --paragraph_file_path /home/ec2-user/SageMaker/repos/LMFlow/data/wa/wa_glossary_question_generation_input.jsonl \
    --output_file_path /home/ec2-user/SageMaker/repos/LMFlow/data/wa/generated_questions/wa_glossary_question.jsonl \
    --model chavinlo/gpt4-x-alpaca \
    --cache_dir /home/ec2-user/SageMaker/repos/LMFlow/cache
"""
import argparse
import json
import logging
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmflow.data_gen import data_util
from lmflow.utils.torch_util import get_device

logging.basicConfig()
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def parse_good_bad(generated_text: str) -> str:
    """
    From the LM output, parse the "good question" from the text
    Used in the gbq format from InPars
    """
    parsed_generated_text = (
        generated_text.split("Example 4:")[1].split("Example 5:")[0].split("Good Question: ")[1]
    )
    if "bad question:" not in parsed_generated_text.lower():
        return ""
    good_question, *_ = re.split("bad question:", parsed_generated_text, flags=re.IGNORECASE)
    good_question = good_question.strip()

    return good_question


def parse_answer(generated_answer: str) -> str:
    """
    From the LM output, parse the "good question" from the text
    Used in the gbq format from InPars
    """
    return generated_answer.split("Answer:")[2].split("\n")[0].strip()


def generate_gbq_prompt(doc_text: str) -> str:
    """
    Generate prompt in gbq format from InPars
    """
    doc1 = (
        "We don't know a lot about the effects of caffeine "
        "during pregnancy on you and your baby. "
        "So it's best to limit the amount you get each day. If you are pregnant, "
        "limit caffeine to 200 milligrams each day. "
        "This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee."
    )

    doc2 = (
        "Passiflora herbertiana. A rare passion fruit native to Australia. "
        "Fruits are green-skinned, white fleshed, with an unknown edible rating. "
        "Some sources list the fruit as edible, sweet and tasty, "
        "while others list the fruits as being bitter and inedible."
    )

    good_question_2 = (
        "What is Passiflora herbertiana (a rare passion fruit) " "and how does it taste like?"
    )

    doc3 = (
        "The Canadian Armed Forces. 1  The first large-scale Canadian "
        "peacekeeping mission started in Egypt on November 24, 1956. 2  "
        "There are approximately 65,000 Regular Force "
        "and 25,000 reservist members in the Canadian military. "
        "3  In Canada, August 9 is designated as National Peacekeepers' Day."
    )

    prompt = f"""Example 1:
Document: {doc1}
Good Question: How much caffeine is ok for a pregnant woman to have?
Bad Question: Is a little caffeine ok during pregnancy?

Example 2:
Document: {doc2}
Good Question: {good_question_2}
Bad Question: What fruit is native to Australia?

Example 3:
Document: {doc3}
Good Question: Information on the Canadian Armed Forces size and history.
Bad Question: How large is the Canadian military?

Example 4:
Document: {doc_text}
Good Question:"""

    return prompt


def generate_answer_for_question_prompt(question, doc):
    prompt = f"""
    What is the answer to question according to document.
    Question: What is an appraiser?
    Document: "Each resident of a cooperative apartment unit pays a monthly maintenance fee, similar to the condo association fees paid by condominium owners.
Cooperatives can be more attractive than renting an apartment unit for some people, particularly in metropolitan areas where the cost of living is higher.
Some cooperatives cater specifically to certain types of residents, like senior citizens.
Both condo owners and cooperative residents are generally responsible for maintaining the interior of their individual units, and are responsible for paying their own utility costs.
Here is an example of how you might encounter cooperatives when you’re working as a real estate professional: Pat is a real estate developer who decided to build a 50-unit cooperative building for senior citizens."
    Answer: Cooperatives residents are responsible for maintaining the interior of their individual units, and are responsible for paying their own utility costs.
    
    Question: {question}
    Document: "{doc}"
    Answer:"""
    return prompt


def main():
    parser = argparse.ArgumentParser(description="InPars Query generation")
    parser.add_argument(
        "--paragraph_file_path",
        type=str,
        help="Path to the paragraph data jsonl file",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="LM model name or path",
        required=False,
        default="EleutherAI/gpt-j-6B",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="location for cached model artifact",
        required=False,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="Number of new tokens to generate. 50-100 is good",
        required=False,
        default=75,
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        help="Output Directory",
        required=True,
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--load_in_8bits", action="store_true", default=False)

    args = parser.parse_args()
    device = get_device()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    logger.debug(f"torch version: {torch.__version__}")
    logger.debug(f"CUDA arch list: {torch.cuda.get_arch_list()}")

    # get max memory
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
    max_memory = f"{free_in_GB - 2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    logger.info(f"Max cuda memory: {max_memory}")

    # load model + tokenizer
    # default to fp16, or 8 bit quantization
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if args.load_in_8bits:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", load_in_8bit=True, max_memory=max_memory,
            cache_dir=args.cache_dir
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=torch.float16, cache_dir=args.cache_dir
        )

    # Get some numbers for controlling generation
    MAX_NEW_TOKENS = args.max_new_tokens
    MAX_CONTEXT_LENGTH = model.config.n_positions

    # load paragraph data
    paragraph_data = data_util.from_jsonl(args.paragraph_file_path)

    # create output directory if not there
    output_folder = os.path.dirname(args.output_file_path)
    os.makedirs(output_folder, exist_ok=True)

    # generate synthetic data
    logger.info("Generating synthetic data")
    instruction_data = []
    with open(os.path.join(output_folder, "generated_qa_doc.jsonl"), "w") as outfile:
        for idx, d in enumerate(tqdm(paragraph_data, miniters=int(len(paragraph_data) / 10))):
            par_text = d["text"]
            # par_id = d["par_id"]
            # sentence_id = d["sentence_id"]

            # generate context for input
            prompt = generate_gbq_prompt(par_text)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # skip cases where paragraphs are too long
            if inputs["input_ids"].shape[1] + MAX_NEW_TOKENS > MAX_CONTEXT_LENGTH:
                continue

            greedy_output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            generated_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

            # parse the good_q from generated_text
            # if good_q not generated, skip
            try:
                good_q = parse_good_bad(generated_text)
            except ValueError:
                logger.exception(f"broke, generated text at idx {idx} is: \n {generated_text}")
                break
            if not good_q:
                continue

            answer_prompt = generate_answer_for_question_prompt(good_q, par_text)

            inputs = tokenizer(answer_prompt, return_tensors="pt").to(device)

            # skip cases where paragraphs are too long
            if inputs["input_ids"].shape[1] + MAX_NEW_TOKENS > MAX_CONTEXT_LENGTH:
                continue

            greedy_output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            generated_answer = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

            # parse the good_q from generated_text
            # if good_q not generated, skip
            try:
                answer = parse_answer(generated_answer)
            except ValueError:
                logger.exception(f"broke, generated text at idx {idx} is: \n {generated_text}")
                break
            if not good_q:
                continue

            # query_id = f"{par_id}_query"
            instruction_dict = {
                "input": good_q,
                "output": answer,
            }

            instruction_data.append(instruction_dict)
            json.dump({
                "question": good_q,
                "doc": par_text,
                "answer": answer
            }, outfile)
            outfile.write("\n")
            outfile.flush()

    assert len(instruction_data) <= len(paragraph_data)

    logger.info(f"Generated {len(instruction_data)} queries from {len(paragraph_data)} paragraphs")

    lmflow_data = {
        "type": "text2text",
        "instances": instruction_data
    }
    data_util.write_json(args.output_file_path, lmflow_data)
    logger.info(f"Synthetic data written to {args.output_file_path}")


if __name__ == '__main__':
    main()
