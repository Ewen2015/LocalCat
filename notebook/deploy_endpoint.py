
import boto3  
from sagemaker.huggingface.model import HuggingFaceModel
import sagemaker
from time import gmtime, strftime

# Model Location
config = dict()
config["S3_MODEL"] = "s3://hugging-face/llm/chatglm2-6b-int4.tar.gz"
config["MODEL_NAME"] = "chatglm2-6b-int4"

# Role
try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    config["ROLE"] = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Configuration
huggingface_model = HuggingFaceModel(
    model_data=config["S3_MODEL"],
    role=config["ROLE"],
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
)

# Deployment
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.4xlarge',
    endpoint_name=config['MODEL_NAME'].upper() + strftime("-%Y%m%d-%H%M%S", gmtime()),
)

## Envoke
from sagemaker.huggingface.model import HuggingFacePredictor

predictor = HuggingFacePredictor(
    endpoint_name='CHATGLM2-6B-INT4-20231204-030044'
)