import os 
os.execute("pip install einops")

import torch
import transformers
import torch.nn.functional as F

model_name = "falcon-7b"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

def to_embeddings(model,text):
    input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
    model_output = model(input_ids, output_hidden_states=True)
    data = (model_output.hidden_states[-1].transpose(0, 1))[0]
    data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)
    return data.tolist()

def model_fn(model_dir):
    pipe = transformers.AutoModel.from_pretrained(model_name,trust_remote_code=True).half()
    pipe.to("cuda")
    return pipe


def predict_fn(data, pipe):
    text = data.pop("text", data)
    type = data.pop("type", 0)

    if type == 0:
        response, history = pipe.chat(tokenizer, text, history=[])
        return response
    else:
        return to_embeddings(pipe, text)