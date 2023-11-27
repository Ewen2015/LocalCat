# LocalCat

LocalCat enables users to deploy large language models (LLMs) from Hugging Face to local environments, such as personal laptops or cloud platforms like AWS.

## GLM

- Model: [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
- Model: [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
- Model: [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b)

## Model Download

- Step1: Install `git-lfs` [here](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing)
- Step2: Clone model repo.
- Step3: Play with your models.

### Step 1
#### Mac
```bash
brew install git-lfs
```

#### Linux
```bash
# yum/rpm repos: 
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
sudo yum install git-lfs

# apt/deb repos: 
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

### Step 2
```bash
git clone https://github.com/THUDM/ChatGLM3.git

cd ChatGLM3 
pip install -r requirements.txt
```
