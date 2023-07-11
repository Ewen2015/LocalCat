#!/bin/bash

rm chatglm2-6b-int4.tar.gz
tar zcvf chatglm2-6b-int4.tar.gz *

aws s3 cp chatglm2-6b-int4.tar.gz \
  s3://hugging-face/llm/chatglm2-6b-int4.tar.gz
