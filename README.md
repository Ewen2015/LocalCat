# LocalCat

LocalCat enables users to deploy large language models (LLMs) from Hugging Face to local environments, such as personal laptops or cloud platforms like AWS.

## GLM

- Model: [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
- Model: [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
- Model: [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b)

## Model Download

### HuggingFace

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device=device)
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
```

```bash
cd ~/.cache/huggingface/hub/
```

### `git-lfs`
- Step1: Install `git-lfs` [here](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing)
- Step2: Clone model repo.
- Step3: Play with your models.

<details><summary>Details</summary>
#### Step 1
##### Mac
```bash
brew install git-lfs
```

##### Linux
```bash
# yum/rpm repos: 
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
sudo yum install git-lfs

# apt/deb repos: 
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

#### Step 2
```bash
git clone https://github.com/THUDM/ChatGLM3.git

cd ChatGLM3 
pip install -r requirements.txt
```
</details>

## Deploy on AWS
Go to [code](notebook/deploy_endpoint.py) for more details.

You should store your LLM on `s3` and use `sagemaker.huggingface` to deploy and envoke the your model.

<details><summary>Code</summary>

```python
from sagemaker.huggingface.model import HuggingFaceModel

# Configuration
huggingface_model = HuggingFaceModel(
    model_data=config["S3_MODEL"],
    role=config["ROLE"],
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
)

# Deployment
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.4xlarge',
    endpoint_name=config['MODEL_NAME'].upper() + strftime("-%Y%m%d-%H%M%S", gmtime()),
)
```

</details>

## LangChain
[LangChain Official Guide](https://python.langchain.com/docs/integrations/llms/sagemaker) to connect AWS Endpoint and LangChain.

<details><summary>Code</summary>

```python
class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({prompt: prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        # return response_json[0]["generated_text"]
        return response_json

content_handler = ContentHandler()

llm = SagemakerEndpoint(
         endpoint_name=ENDPOINT_NAME, 
         region_name=REGION, 
         model_kwargs={"max_new_tokens": 700, "top_p": 0.9, "temperature": 0.6},
         endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
         content_handler=content_handler
     )
```
</details>

### LangChain Expression Language (LCEL)
Basic example: prompt + model + output parser

```python
chain = prompt | model | output_parser
```
<details><summary>Use case: manufacture diagnostic trouble analysis</summary>

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

model = llm
prompt_template = """
tell me a possible cause of the diagnostic trouble {trouble} in the process of car-making"
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

print(chain.invoke({"trouble": "battery"}))
```
</details>

### Retrieval Augmented Generation (RAG)

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```



<details><summary>Use case</summary>

```python
# Get the Document
import requests
from bs4 import BeautifulSoup

# url = "https://en.wikipedia.org/wiki/GPT-4"
url = "https://openai.com/research/gpt-4"
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')

# find the content div
content_div = soup.find('div', {'class': 'ui-block--text'})

# remove unwanted elements from div
unwanted_tags = ['sup', 'span', 'table', 'ul', 'ol']
for tag in unwanted_tags:
    for match in content_div.findAll(tag):
        match.extract()

print(content_div.get_text())

# Split the Doc
from langchain.text_splitter import RecursiveCharacterTextSplitter

article_text = content_div.get_text()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
)

texts = text_splitter.create_documents([article_text])

# Embedding Model
from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')

# Vector Store/Vector DB
from langchain.vectorstores import Chroma
# use the text chunks and the embeddings model to fill our vector store
db = Chroma.from_documents(texts, embedding)

from langchain import PromptTemplate
user_question = "ChatGPT and GPT4, which on is more stable?"

# use our vector store to find similar text chunks
results = db.similarity_search(
    query=user_question,
    n_results=5
)

# define the prompt template
template = """
You are a chat bot who loves to help people! Given the following context sections, answer the
question using only the given context. If you are unsure and the answer is not
explicitly writting in the documentation, say "Sorry, I don't know how to help with that."

Context sections:
{context}

Question:
{users_question}

Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "users_question"])

# fill the prompt template
prompt_text = prompt.format(context=results, users_question=user_question)

# ask the defined LLM
llm(prompt_text)
```
</details>