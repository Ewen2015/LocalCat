import torch
import transformers
import torch.nn.functional as F

model_name = "falcon-7b-instruct"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

def predict_fn(data, pipe):
    text = data.pop("text", data)
    type = data.pop("type", 0)

    if type == 0:
        response, history = pipe.chat(tokenizer, text, history=[])
        return response
    else:
        return to_embeddings(pipe, text)