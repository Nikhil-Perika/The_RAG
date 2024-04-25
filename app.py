import streamlit as st
from run import getanswer

prompt=st.text_area("Enter your Query")


if st.button('submit')==True:
    response = getanswer(prompt)

    st.write(response)