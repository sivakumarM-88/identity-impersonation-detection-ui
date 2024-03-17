import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("Welcome 👋")

st.sidebar.success("Select a demo above.")

st.markdown("""Hackathon - Identity impersonation challenge. Please select demo on left panel.
            Audio fake/real analyzer - voiceapp
            Emotion detector - emotapp""");