#!/bin/bash

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm2-6b

cd chatglm2-6b

for ((i=1; i<=7; i++))
do
    filename="pytorch_model-$(printf "%05d" $i)-of-00007.bin"
    rm "$filename"
    url="https://cloud.tsinghua.edu.cn/seafhttp/files/22c6f3eb-6830-4723-851d-f11551b47df5/$filename"
    wget "$url"
done

rm tokenizer.model
wget https://cloud.tsinghua.edu.cn/seafhttp/files/114dde78-0208-40bf-b43a-e941e764c204/tokenizer.model

cd ..