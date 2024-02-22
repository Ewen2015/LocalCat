# from modelscope import snapshot_download
# snapshot_download('ZhipuAI/chatglm3-6b', revision = "v1.0.0", cache_dir="./")

# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id="facebook/mbart-large-50-many-to-one-mmt", filename="pytorch_model.bin")

# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="facebook/mbart-large-50-many-to-one-mmt")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")