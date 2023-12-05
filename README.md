# LocalCat

LocalCat enables users to deploy large language models (LLMs) from Hugging Face to local environments, such as personal laptops or cloud platforms like AWS.

## GLM

- Model: [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
- Model: [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
- Model: [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b)

## Deploy on AWS
Go to [code](notebook/deploy_endpoint.py) for more details.

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

## LangChain
[LangChain Official Guide](https://python.langchain.com/docs/integrations/llms/sagemaker) to connect AWS Endpoint and LangChain.

Go to [code](notebook/test_langchain.py) for more details.



## Model Download

### HuggingFace
```bash
cd ~/.cache/huggingface/hub/
```

### `git-lfs`
- Step1: Install `git-lfs` [here](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing)
- Step2: Clone model repo.
- Step3: Play with your models.

<detials>
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
