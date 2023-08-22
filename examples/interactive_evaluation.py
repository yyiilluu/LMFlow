"""

"""
import json
import os
from string import Template

import deepspeed
import pandas as pd
# import deepspeed
import torch
from tqdm import tqdm
# TODO: remove later
from transformers import AutoConfig
from transformers import HfArgumentParser

from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
from lmflow.models.auto_model import AutoModel
from lmflow.utils.data_utils import set_random_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

model_path = "/home/ec2-user/SageMaker/repos/LMFlow/output_models/pretraining_pythia_1b_ift_compress_decompress_16k"
input_file_path = "/home/ec2-user/SageMaker/repos/compressor/data/nq-dev-processed-pos-only_all_qa_gpt-3.5-turbo-0613.jsonl"
output_file_path = "/home/ec2-user/SageMaker/repos/compressor/data/sample_output.csv"
prompt_template_name = "compression"

PROMPT_TEMPLATES = {
    "compression": """
Compress the following text aggressively without abbreviation, and such that GPT model can still 
answer any question based on this content when used as part of the prompt. Do not need to be human readable, while still keeping ALL the information to fully reconstruct it by GPT model.
```
${content}
```
Compressed text:
""",
    "rec_compression": """
Compress the following text aggressively without abbreviation, and such that you (GPT model) can reconstruct it to the original. Do not need to be human readable. Using language mixing to aggresively compress it is permitted
```
${content}
```
Compressed text:
""",
    "qa_compressed_content": """Content delimited by triple backticks is compressed by GPT model with heavy language mixing. Reconstruct it to original quietly for yourself first and then follow instruction below.
Extract answer to question in short from only the following content
```
${compressed_content}
```

Question: ${question}
Answer:
""",
    "qa_original_content": """Extract answer to question in short from only the following content 
```
${content}
```

Question: ${question}
Answer:
""",
    "summarization": """Summarize the following text with as few tokens as possible, and such that GPT-4 can answer any question based on this content.
```
${content}
```
""",
    "decompression": """Decompress the following text compressed by GPT model aggressively without abbreviation
```
${compressed_content}
```
Original content: 
""",
}

prompt_template = Template(PROMPT_TEMPLATES[prompt_template_name])

class InteractiveEvaluator:
    def __init__(self, model, model_args, evaluator_args):
        self.model = model
        self.evaluator_args = evaluator_args
        self.model_args = model_args
        print("--------Begin Evaluator Arguments----------")
        print(f"model_args : {self.model_args}")
        print(f"evaluator_args : {self.evaluator_args}")
        print("--------End Evaluator Arguments----------")

        # random seed
        set_random_seed(self.evaluator_args.random_seed)
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(self.local_rank)  # NOTE: cpu-only machine will have error
        deepspeed.init_distributed()

        self.config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        try:
            self.model_hidden_size = self.config.hidden_size
        except:
            print("Error in setting hidden size, use the default size 1024")
            self.model_hidden_size = 1024  # gpt2 seems do not have hidden_size in config

        print(f"model_hidden_size = {self.model_hidden_size}")

    def interactive_evaluate(self):
        while True:
            text = input("Enter prompt: ")
            formatted_input = f"Input: {text.strip()}"
            inputs = self.model.encode(formatted_input, return_tensors="pt").to(
                device=self.local_rank)
            with torch.inference_mode():
                outputs = self.model.inference(inputs, max_new_tokens=100, temperature=0.0)
                text_out = self.model.decode(outputs[0], skip_special_tokens=True)
                print("Generated content: \n")
                print(text_out)

    def inference(self, input_file_path):
        prompts = []
        with open(input_file_path) as f:
            for line in f:
                ex = json.loads(line)
                prompts.append(prompt_template.substitute(
                    **{
                        "content": ex['qa_example']['text']
                    }
                ))

        generations = []
        prompts = prompts[:10]
        for prompt in tqdm(prompts):
            inputs = self.model.encode(prompt, return_tensors="pt").to(device=self.local_rank)
            with torch.inference_mode():
                outputs = self.model.inference(inputs, max_new_tokens=100, temperature=0.0)
                generations.append(self.model.decode(outputs[0], skip_special_tokens=True))

        data = []
        for gen, prompt in zip(generations, prompts):
            data.append({
                "prompt": prompt,
                "generation": gen,
            })

        df = pd.DataFrame(data)
        df.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    pipeline_name = "evaluator"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()
    with open(pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = AutoModel.get_model(model_args, tune_strategy='none', ds_config=ds_config)

    interactive_evaluator = InteractiveEvaluator(model, model_args=model_args,
                                                 evaluator_args=pipeline_args)

    # interactive_evaluator.interactive_evaluate()
    interactive_evaluator.inference(input_file_path)

