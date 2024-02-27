#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""

import streamlit as st
import pandas as pd

st.set_page_config(
     page_title="LLM Translator",
     page_icon="💬",
     initial_sidebar_state=st.session_state.get("sidebar_state", "collapsed")
     )

st.title("LLM Translator with AI In-house")

tab1, tab2, tab3 = st.tabs(["Translator", "Fine-tune", "New Request"])
with tab1:
    tab11, tab12 = st.tabs(["Chat", "Batch"])
    with tab11:
        text = st.text_input('Input', 'text to translate')
        st.write("Translation")
        st.code(text)
    with tab12:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            # Read the uploaded file as DataFrame
            df = pd.read_csv(uploaded_file)

            # Display the DataFrame
            st.write("### Uploaded DataFrame")
            st.dataframe(df)

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