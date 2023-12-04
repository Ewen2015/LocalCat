import boto3
from sagemaker.huggingface.model import HuggingFacePredictor
import json
from typing import Dict
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.prompts import PromptTemplate


predictor = HuggingFacePredictor(
    endpoint_name='CHATGLM2-6B-INT4-20231204-030044'
)
ENDPOINT_NAME = predictor.endpoint
session = boto3.session.Session()
REGION = session.region_name

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({prompt: prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        # return response_json[0]["generated_text"]
        return response_json

content_handler = ContentHandler()

llm = SagemakerEndpoint(
         endpoint_name=ENDPOINT_NAME, 
         region_name=REGION, 
         model_kwargs={"max_new_tokens": 700, "top_p": 0.9, "temperature": 0.6},
         endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
         content_handler=content_handler
     )