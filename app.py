import streamlit as st

# Main entry point
st.set_page_config(page_title="Q&A App", page_icon="❓", layout="centered")

st.title("Welcome to the Q&A App")
st.write("This is the main entry point. Use the left sidebar to navigate to different sections of the app.")
st.sidebar.title("Navigation")
st.sidebar.write("Select a page to navigate")
