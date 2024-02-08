import torch
import transformers
import torch.nn.functional as F

model_name = "chatglm2-6b-int4"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def model_fn(model_dir):
    pipe = transformers.AutoModel.from_pretrained(model_name, trust_remote_code=True).to("cuda")
    return pipe

def predict_fn(data, pipe):
    input = data.pop("input", data)

    response, history = pipe.chat(tokenizer, input, history=[])
    return response