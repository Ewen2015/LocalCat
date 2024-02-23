#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""

import os
import json
from time import gmtime, strftime
import subprocess
import pandas as pd
from tqdm import tqdm
import boto3
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.huggingface.model import HuggingFacePredictor

pd.set_option('display.max_colwidth', None)

class Local:
    def __init__(self, model_path=None, model_name=None):
        
        self.model_path = model_path
        self.model_name = model_name
        
    
    def push_to_s3(self, bucket, prefix=None):
        current_dir = os.getcwd()
        
        file_tar = f"{self.model_name}.tar.gz"
        key = f"{file_tar}" if prefix is None else f"{prefix}/{file_tar}"
        self.s3_model = f"s3://{bucket}/{key}"

        # Define the bash command
        bash_command = f"""
        cd {self.model_path}
        tar zcvf {file_tar} *
        aws s3 cp {file_tar} {self.s3_model}
        rm {file_tar}
        cd {current_dir}
        """

        # Run the bash command
        process = subprocess.Popen(bash_command, shell=True)
        process.wait()
        return None
    
    def deploy(self, instance_type='ml.g4dn.4xlarge', 
               transformers_version='4.37.0', pytorch_version='2.1.0', py_version='py310',):
        try:
            self.role = sagemaker.get_execution_role()
        except ValueError:
            iam = boto3.client('iam')
            self.role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
        
        huggingface_model = HuggingFaceModel(
            model_data=self.s3_model,
            role=self.role,
            transformers_version=transformers_version,
            pytorch_version=pytorch_version,
            py_version=py_version,
        )
        
        self.endpoint_name = self.model_name.upper() + strftime("-%Y%m%d-%H%M%S", gmtime())
        
        self.predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=self.endpoint_name,
        )
        
        return None
        
    def translator(self, text):
        predictor = HuggingFacePredictor(
            endpoint_name=self.endpoint_name
        )
        runtime_client = boto3.client('sagemaker-runtime')
        input_data = {"inputs": text}

        response = runtime_client.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps(input_data)
                )
        result = json.loads(response['Body'].read().decode('utf-8'))[0]['generated_text']
        return result
    
    def translator_batch(self, df, col_src='Chinese', col_tgt='English'):
        tqdm.pandas()
        df[col_tgt] = df[col_src].progress_apply(lambda x: self.translator(x))
        return df