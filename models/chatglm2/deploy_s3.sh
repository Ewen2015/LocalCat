#!/bin/bash

rm chatglm2-6b.tar.gz
tar zcvf chatglm2-6b.tar.gz *

aws s3 cp chatglm2-6b.tar.gz \
  s3://hugging-face/llm/chatglm2-6b.tar.gz
