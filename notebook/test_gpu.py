from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)