#!/bin/bash

rm chatglm3-6b.tar.gz
tar zcvf chatglm3-6b.tar.gz *

aws s3 cp chatglm3-6b.tar.gz \
  s3://hugging-face/llm/chatglm3-6b.tar.gz
