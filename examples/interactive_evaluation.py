"""

"""
import json
import os

import deepspeed
# import deepspeed
import torch
# TODO: remove later
from transformers import AutoConfig
from transformers import HfArgumentParser

from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
from lmflow.models.auto_model import AutoModel
from lmflow.utils.data_utils import set_random_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers


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
            inputs = self.model.encode(formatted_input, return_tensors="pt").to(device=self.local_rank)
            with torch.inference_mode():
                outputs = self.model.inference(inputs, max_new_tokens=100, temperature=0.0)
                text_out = self.model.decode(outputs[0], skip_special_tokens=True)
                print("Generated content: \n")
                print(text_out)


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

    interactive_evaluator.interactive_evaluate()
