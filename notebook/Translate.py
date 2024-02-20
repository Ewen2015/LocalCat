#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translate:
    """
    Translate class for language translation.
    """
    def __init__(self, 
                 model_name_or_path='facebook/mbart-large-50-many-to-one-mmt', 
                 src_lang='zh_CN', 
                 tgt_lang='en_XX'):
        """
        Initializes the Translate class.

        Args: 
            model_name_or_path (str): Name or path to the pretrained model and tokenizer. Default is 'facebook/mbart-large-50-many-to-one-mmt'. 
            src_lang (str): Source language for translation. Default is 'zh_CN'. 
            tgt_lang (str): Target language for translation. Default is 'en_XX'.
        """

        self.model_name_or_path = model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

    def translator(self, text, max_new_tokens=500):
        """
        Translates text from source language to target language.

        Args:
            text (str): Text to be translated.
            max_new_tokens (int): The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. Default is 500.

        Returns:
            str: Translated text.
        """

        encoded_text = self.tokenizer(text, return_tensors="pt").to(self.device)
        translated_output = self.model.generate(**encoded_text, max_new_tokens=max_new_tokens)
        translated_text = self.tokenizer.decode(translated_output[0], skip_special_tokens=True)

        return translated_text