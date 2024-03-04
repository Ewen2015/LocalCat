Tutorials
*********

This tutorial will guide you through the process of using **LocalCat** to load pre-trained LLMs and fine-tune them with domain data.

Load Pre-trained LLMs
=====================

**LocalCat** simplifies the process of loading pre-trained LLMs.

.. code-block:: Python
    
    from LocalCat.Translate import Translate

    model_mbart = "facebook/mbart-large-50-many-to-many-mmt"

    trans = Translate(model_name_or_path=model_mbart,
                      src_lang='zh_CN', 
                      tgt_lang='en_XX')

.. note::

    For developers in China, you may use hugging face mirror sites, such as `hf-mirror.com <https://hf-mirror.com/>`_. 
    
    .. code-block:: bash

        HF_ENDPOINT=https://hf-mirror.com python3 download_model.py

Translate by Case
-----------------

.. code-block:: Python

    text = "开空调的情况下，续航掉的太快了，特别是冬天天气冷的时候，不开空调不行，天气一冻，续航就掉的更快"
    print(trans.translator(text=text))

    # If the air conditioner is on, the flight resumes too quickly, especially in winter when the weather is cold. If the air conditioner is not on, the flight resumes faster as soon as the weather is cold

Translate on Batch
------------------

**LocalCat** also supports batch translation. All you need to do is to load the data and call the `translator_batch` method.

The data should be in the form of a `pandas` DataFrame.

.. code-block:: Python
    
    import pandas as pd
    
    file_inference = "DATA_INFERENCE_2024.csv"

    df = pd.read_csv(file_inference)
    df = trans.translator_batch(df=df, 
                                col_src='Chinese', 
                                col_tgt="Translation")

Fine-tune LLMs with Domain Data
===============================

**LocalCat** also supports fine-tuning pre-trained LLMs with domain data.

Fine-tune LLM 
-------------

.. code-block:: Python

    file_training = "DATA_TRIAN_2023.csv"
    finetuned_model_path = "mbart-finetuned-cn-to-en-auto"

    df = pd.read_csv(file_training)

    trans = Translate(model_mbart)
    trans.finetune(df=df, 
                   finetuned_model_path=finetuned_model_path,
                   train_size=0.9, 
                   batch_size=4,
                   learning_rate=2e-5,
                   num_train_epochs=1)

Load and Test the Fine-tuned Model
----------------------------------

.. code-block:: Python
    
    model_finetuned = "mbart-finetuned-cn-to-en-auto"

    text = "开空调的情况下，续航掉的太快了，特别是冬天天气冷的时候，不开空调不行，天气一冻，续航就掉的更快"

    trans = Translate(model_name_or_path=model_finetuned)
    print(trans.translator(text=text))

The following is a comparison of the translation results before and after fine-tuning: 

.. epigraph::
    
    **MBart:** If the air conditioner is on, the *flight resumes* too quickly, especially in winter when the weather is cold. If the air conditioner is not on, the *flight resumes* faster as soon as the weather is cold.
    
    **Fine-tuned:** In the case of turning on the air conditioner, the *electric range* drops too fast, especially when the weather is cold in winter, not turning on the air conditioner is not possible, the weather freezes, and the *electric range* drops faster.

Deploy the LLM
==============

**LocalCat** also supports deploying LLMs on the cloud (AWS Sagemaker Endpoint). Deploying the model contains the following steps:

1. Push the model to S3
2. Deploy the model as an endpoint
3. Test the endpoint

.. code-block:: Python
    
    from LocalCat.Translate import Local

    model_path = "../models/"
    model_finetuned = "mbart-finetuned-cn-to-en-auto"

    # 1. Push the model to S3
    bucket = "ai"
    prefix = "llm"

    local = Local(model_name=model_finetuned, model_path=model_path)
    local.push_to_s3(bucket=bucket, prefix=prefix)

    # 2. Deploy the model as an endpoint
    local.deploy(instance_type='ml.g4dn.4xlarge',
                 transformers_version='4.37.0', 
                 pytorch_version='2.1.0', 
                 py_version='py310')

    # 3. Test the endpoint
    # Check the endpoint name in the AWS Sagemaker Console
    local = Local()
    local.endpoint_name = "MBART-20240226-024324" 

    text = "开空调的情况下，续航掉的太快了，特别是冬天天气冷的时候，不开空调不行，天气一冻，续航就掉的更快"
    result = local.translator(text=text)
    print(result)
    # In the case of turning on the air conditioner, the electric range drops too fast, especially when the weather is cold in winter, not turning on the air conditioner is not possible, the weather freezes, and the electric range drops faster.