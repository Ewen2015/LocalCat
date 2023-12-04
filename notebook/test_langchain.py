from sagemaker.huggingface.model import HuggingFacePredictor

predictor = HuggingFacePredictor(
    endpoint_name='CHATGLM2-6B-INT4-20231204-030044'
)

ENDPOINT_NAME = predictor.endpoint

import boto3
session = boto3.session.Session()
REGION = session.region_name


from langchain.docstore.document import Document

example_doc_1 = """
Peter and Elizabeth took a taxi to attend the night party in the city. While in the party, Elizabeth collapsed and was rushed to the hospital.
Since she was diagnosed with a brain injury, the doctor told Peter to stay besides her until she gets well.
Therefore, Peter stayed with her at the hospital for 3 days without leaving.
"""

docs = [
    Document(
        page_content=example_doc_1,
    )
]

import json
from typing import Dict

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.prompts import PromptTemplate

query = """How long was Elizabeth hospitalized?
"""

prompt_template = """Use the following pieces of context to answer the question at the end.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

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


chain = load_qa_chain(
    llm=SagemakerEndpoint(
         endpoint_name=ENDPOINT_NAME, 
         region_name=REGION, 
         model_kwargs={"max_new_tokens": 700, "top_p": 0.9, "temperature": 0.6},
         endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
         content_handler=content_handler
     ),
    prompt=PROMPT,
)

print(chain({"input_documents": docs, "question": query}, return_only_outputs=True))