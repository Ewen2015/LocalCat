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

model_path = "../models/"
image_path = "../images/"

model_name = "mbart-finetuned-cn-to-en-auto"
image_name = "finetune.png"

@st.cache_resource(ttl="1h", show_spinner="Loading model...")
def load_model(model_path, model_name):
    return Translate(model_path+model_name)

st.title("✨ Translator Powered by GenAI")
st.write("The first GenAI powered product is here!")
st.write("This is a translator based on the Open Source **Large Language Model (LLM)**. It is a powerful model that can accurately translate between Chinese and English in the **auto industry scenario**. It can also be fine-tuned using your own data.")

tab1, tab2, tab3 = st.tabs(["Translator", "Fine-tune", "New Request"])
with tab1:
    st.write("Here, you can translate your text by case or on batch.")
    tab11, tab12 = st.tabs(["One by One", "Batch"])
    with tab11:
        st.write("###### Text to translate")
        text = st.text_area(label="Chinese 中文", value=None)
        if text is not None:
            trans = load_model(model_path, model_name)
            with st.spinner(text="Translating. This may take a moment..."):
                result = trans.translator(text)
        else:
            result = "You will see the translation here."
        st.write("###### Translation")
        st.text_area(label="English", value=result)
    with tab12:
        st.write("Your data should contains the column `Chinese`, which is source language you try to translate.")
        uploaded_file = st.file_uploader("Choose a CSV file to translate", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### Data to Translate")
            st.dataframe(df)

            if st.button('Translate'):
                trans = load_model(model_path, model_name)

                with st.spinner(text="Translating. Take a coffee break.☕️"):
                    df = trans.translator_batch(df, col_tgt="Translation")
                
                    st.write("### Translation Results")
                    st.dataframe(df)
                    st.download_button(
                        label="Download",
                        data=df.to_csv().encode('utf-8'),
                        file_name='LLM_TRANSLATOR_RESULTS.csv',
                        mime='text/csv',
                    )

with tab2:
    st.write("Here, you can fine-tune the model with your own data.")
    try:
        st.image(image_path+image_name, caption='LLM Fine-tunes (Source: https://abacus.ai/llmapi)')
    except:
        pass
    st.write("We've loaded the pre-trained LLM for your. All you need to do in **Fine-tuning a LLM** is to find **labeled examples** and upload them.")
    st.write("Your data should contains two columns: `Chinese` and `English`. Each row is a new example for the machine to learn (so-called **machine learning**).")
    
    uploaded_file = st.file_uploader("Choose a CSV file to fine-tune", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data for Fine-tuning")
        st.dataframe(df)

        if st.button('Fine-tune'):
            st.write("☕️ Fine-tuning the model. This may take a while...")

with tab3:
    st.write("Here, you can make requests to the translator based on the **latest terminology**.")
    st.write("Please share your insights and suggestions.")
    st.write("See the following two as examples.")
    st.write("")

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