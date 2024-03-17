import streamlit as st

import classifyPage

st.set_page_config(
    page_title="Hackathon impersonation tool for images",
    page_icon="🤖",
    layout="wide")

PAGES = {
    
    "Classify Image": classifyPage
}

st.sidebar.title("WF Hackathon tool for identifying impersonated images")

st.sidebar.write("ours is a tool that utilizes the power of Deep Learning to distinguish Real images from the Fake ones.")

st.sidebar.subheader('Navigation:')
selection = st.sidebar.radio("", list(PAGES.keys()))

page = PAGES[selection]

page.app()