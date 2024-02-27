#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""

import streamlit as st
import pandas as pd
from LocalCat.Translate import Translate

st.set_page_config(
     page_title="LLM Translator",
     page_icon="💬",
     initial_sidebar_state=st.session_state.get("sidebar_state", "collapsed")
     )

st.title("LLM Translator with AI In-house")

tab1, tab2, tab3 = st.tabs(["Translator", "Fine-tune", "New Request"])
with tab1:

    try:
        model_name = f"mbart-finetuned-cn-to-en-auto"
        model_path = f"../models/{model_name}"
        st.info("Loading the model. This can take a while.")
        trans = Translate(model_path)
        model_ready = True
    except:
        st.warnings("The model is not available. Please check the model path.")
        model_ready = False

    tab11, tab12 = st.tabs(["Chat", "Batch"])
    with tab11:
        text = st.text_input('Input')
        if text is not None:
            if model_ready:
                result = trans.translator(text)
            else:
                result = "The model is not available. Please check the model path."
        st.write("Translation")
        st.code(result)
    with tab12:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if model_ready:
                if st.button('Translate!'):
                    df = trans.translator_batch(df, col_tgt="Translation")
            else:
                st.warnings("The model is not available. Please check the model path.")
            
            st.write("### DataFrame")
            st.dataframe(df)
            st.download_button(
                label="Download data as CSV",
                data=df.to_csv().encode('utf-8'),
                file_name='LLM_TRANSLATOR_RESULTS.csv',
                mime='text/csv',
            )

with tab2:
    st.write("This section allows users to fine-tune the model with their own data.")
    st.write("We are working on it. Please stay tuned.📬")

with tab3:
    st.write("This section allows users to request new features for translation.")
    st.write("We are working on it. Please stay tuned.📬")

    df = pd.DataFrame(
        [
        {"Word": "窝窝", "Chinese": "我非常喜欢我的窝窝!", "English": "I like my Volvo so much!"},
        {"Word": "小窝", "Chinese": "有时候小窝会自己出来。", "English": "Sometimes the voice assistant will come out by itself."},
        {"Word": "", "Chinese": "", "English": ""},
        ]
    )
    edited_df = st.data_editor(df)

with st.sidebar:
    st.header("About")
    st.write("This is project is powered by [LocalCat](https://localcat.readthedocs.io).")
    st.markdown("**LocalCat** is designed to make the LLM strategy easy to implement. It provides a simple API and an intuitive interface for loading, fine-tuning, and deploying large deep learning models. With LocalCat, you can easily fine-tune pre-trained models on your own data and leverage their power for your specific tasks.")