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

model_name = f"mbart-finetuned-cn-to-en-auto"
model_path = f"../models/{model_name}"

st.title("LLM Translator with AI In-house")

with st.expander("Pain Points to Solve"):
    st.write("""
    1. High cost of translation services. (1 RMB/item and we have a large and increasing customer base in China)
    2. Inaccurate translation in industrial and brand-specific scenarios. For instance, '窝窝' is a nickname for Volvo Cars, coined by Volvo customers in China.
    3. Difficulty transferring knowledge when transitioning to a new translation service.
    """)

tab1, tab2, tab3 = st.tabs(["Translator", "Fine-tune", "New Request"])
with tab1:
    st.write("Here, you can translate your text by case or on batch.")
    tab11, tab12 = st.tabs(["Chat", "Batch"])
    with tab11:
        st.write("###### Text to translate (Chinese)")
        text = st.text_input(label="Input", value=None)
        if text is not None:
            st.info("Loading the model and translating. Take a moment.")
            trans = Translate(model_path)
            result = trans.translator(text)
        else:
            result = "You will see the translation here."
        st.write("###### Translation (English)")
        st.code(result)
    with tab12:
        uploaded_file = st.file_uploader("Choose a CSV file to translate", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### Data to Translate")
            st.dataframe(df)

            if st.button('Translate'):
                st.info(f"Loading the model. Take a coffee break.☕️")
                trans = Translate(model_path)
                st.info(f"Translating. Take a coffee break.☕️")
                df = trans.translator_batch(df, col_tgt="Translation")
            
                st.write("### Translation Results")
                st.dataframe(df)
                st.download_button(
                    label="Download data as CSV",
                    data=df.to_csv().encode('utf-8'),
                    file_name='LLM_TRANSLATOR_RESULTS.csv',
                    mime='text/csv',
                )

with tab2:
    st.write("Here, you can fine-tune the model with your own data.")
    st.image('finetune.png', caption='LLM Fine-tunes (Source: https://abacus.ai/llmapi)')
    st.write("Your data should contains two columns: `Chinese` and `English`. Each row is a new example for the machine to learn (so-called **machine learning**).")
    
    uploaded_file = st.file_uploader("Choose a CSV file to fine-tune", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data for Fine-tuning")
        st.dataframe(df)

with tab3:
    st.write("Here, you can make requests to the translator based on the **latest terminology**.")
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