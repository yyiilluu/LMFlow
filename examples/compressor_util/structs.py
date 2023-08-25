from collections import namedtuple
from typing import List, Any, Dict

from pydantic import BaseModel

OpenAiResult = namedtuple("OpenAiResult", ["text", "logprobs"])

import tiktoken

text = """"
Compress the following text aggressively without abbreviation, and such that GPT model can still 
answer any question based on this content when used as part of the prompt. Do not need to be human readable, while still keeping ALL the information to fully reconstruct it by GPT model.
```
Does He Love You "Does He Love You" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album "Greatest Hits Volume Two". It is one of country music's several songs about a love triangle. "Does He Love You" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members
```
Compressed text:
"""


def count_token(t):
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(tokenizer.encode(t))


class QAExample(BaseModel):
    question: str
    text: str
    answers: List[str]
    has_answer: bool
    compressed_text: str = ""

    text_len: int = 0
    compressed_text_len: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_len = count_token(self.text)
        self.compressed_text_len = count_token(self.compressed_text)

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]):
        return cls(
            question=dict["question"],
            text=dict["text"],
            answers=dict["answers"],
            has_answer=dict["has_answer"],
            compressed_text=dict["compressed_text"],
            text_len=dict.get("text_len", count_token(dict["text"])),
            compressed_text_len=dict.get(
                "compressed_text_len", count_token(dict["compressed_text"])
            ),
        )
