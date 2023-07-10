cd ./src

rm falcon-7b-instruct.tar.gz
tar zcvf falcon-7b-instruct.tar.gz *

aws s3 cp falcon-7b-instruct.tar.gz \
  s3://hugging-face/llm/falcon-7b-instruct.tar.gz
