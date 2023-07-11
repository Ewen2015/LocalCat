import torch
import transformers
import torch.nn.functional as F

model_name = "falcon-7b-instruct"

<<<<<<< HEAD
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
=======
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def model_fn(model_dir):
    pipe = transformers.AutoModel.from_pretrained(model_name, trust_remote_code=True).half()
    pipe.to("cuda")
    return pipe
>>>>>>> 75bad01e4501fa0561143241b23527f7ca3b928e

def predict_fn(data, pipe):
    input = data.pop("input", data)

    response, history = pipe.chat(tokenizer, input, history=[])
    return response