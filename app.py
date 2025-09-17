# app.py
import streamlit as st
from streamlit.web import cli as stcli
import sys

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "pages/1_Home.py"]
    sys.exit(stcli.main())
