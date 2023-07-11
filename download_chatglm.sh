#!/bin/bash

for ((i=1; i<=7; i++))
do
    filename="pytorch_model-$(printf "%05d" $i)-of-00007.bin"
    url="https://cloud.tsinghua.edu.cn/seafhttp/files/22c6f3eb-6830-4723-851d-f11551b47df5/$filename"
    wget "$url"
done