cd ./src

rm chatglm-6b.tar.gz
tar zcvf chatglm-6b.tar.gz *

aws s3 cp chatglm-6b.tar.gz \
  s3://hugging-face/llm/chatglm-6b.tar.gz
