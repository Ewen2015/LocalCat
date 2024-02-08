#!/bin/bash

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm2-6b-int4

cd chatglm2-6b-int4

rm pytorch_model.bin
wget https://cloud.tsinghua.edu.cn/seafhttp/files/8d97af80-bada-4e1d-9ced-04eeec7141d8/pytorch_model.bin

rm tokenizer.model
wget https://cloud.tsinghua.edu.cn/seafhttp/files/a00be86a-315c-469f-8b59-7014b30804db/tokenizer.model

cd ..