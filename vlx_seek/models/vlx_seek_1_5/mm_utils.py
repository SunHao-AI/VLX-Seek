import torch
from transformers import StoppingCriteria

from .constants import DEFAULT_OBJECT_INDEX, IMAGE_TOKEN_INDEX


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]
    input_ids = []
    offset = 0
    if (
        prompt_chunks
        and prompt_chunks[0]
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for chunk_index, chunk in enumerate(prompt_chunks):
        input_ids.extend(chunk[offset:])
        if chunk_index < len(prompt_chunks) - 1:
            input_ids.extend([image_token_index] * (offset + 1))

    if return_tensors is None:
        return input_ids
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    raise ValueError(f"Unsupported tensor type: {return_tensors}")


def tokenizer_image_object_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    object_token_index=DEFAULT_OBJECT_INDEX,
    return_tensors=None,
):
    prompt_chunks = [
        [tokenizer(chunk).input_ids for chunk in image_chunk.split("<objfeat>")]
        for image_chunk in prompt.split("<image>")
    ]
    input_ids = []
    offset = 0
    if (
        prompt_chunks
        and prompt_chunks[0]
        and prompt_chunks[0][0]
        and prompt_chunks[0][0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0][0])

    for image_index, chunk_group in enumerate(prompt_chunks):
        input_ids.extend(chunk_group[0][offset:])
        for chunk in chunk_group[1:]:
            input_ids.append(object_token_index)
            input_ids.extend(chunk)
        if image_index < len(prompt_chunks) - 1:
            input_ids.append(image_token_index)

    if return_tensors is None:
        return input_ids
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    raise ValueError(f"Unsupported tensor type: {return_tensors}")


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            keyword_ids = tokenizer(keyword).input_ids
            if len(keyword_ids) > 1 and keyword_ids[0] == tokenizer.bos_token_id:
                keyword_ids = keyword_ids[1:]
            self.max_keyword_len = max(self.max_keyword_len, len(keyword_ids))
            self.keyword_ids.append(torch.tensor(keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        for keyword_id in self.keyword_ids:
            keyword_id = keyword_id.to(output_ids.device)
            if torch.equal(output_ids[0, -keyword_id.shape[0] :], keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        return any(keyword in outputs for keyword in self.keywords)

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return all(
            self.call_for_batch(output_ids[index].unsqueeze(0), scores)
            for index in range(output_ids.shape[0])
        )
