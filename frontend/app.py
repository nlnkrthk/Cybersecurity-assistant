"""
Streamlit front end of the Cybersecurity Incident Response Assistant.

    python scripts/frontend/run_frontend.py      (or: streamlit run frontend/app.py from the project root)

A thin client: every page calls the FastAPI backend (API_BASE_URL, default
http://localhost:8000) through api_client.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from components import inject_styles, status_sidebar  # noqa: E402

st.set_page_config(page_title="Incident Response Assistant", page_icon=":material/shield:", layout="wide")

VIEWS = Path(__file__).resolve().parent / "views"
PAGES = [
    st.Page(VIEWS / "home.py", title="Home", icon=":material/home:", default=True),
    st.Page(VIEWS / "assistant.py", title="AI Assistant", icon=":material/smart_toy:"),
    st.Page(VIEWS / "flow_analyzer.py", title="Flow Analysis", icon=":material/query_stats:"),
    st.Page(VIEWS / "incident_search.py", title="Incident Search", icon=":material/manage_search:"),
    st.Page(VIEWS / "triage_queue.py", title="Triage Queue", icon=":material/format_list_numbered:"),
    st.Page(VIEWS / "answer_quality.py", title="Quality & Feedback", icon=":material/fact_check:"),
]

inject_styles()
navigation = st.navigation(PAGES, position="top")
with st.sidebar:
    st.markdown("### :material/shield: Incident Response")
    st.caption("AI-assisted investigation of network security incidents (CIC-IDS2017).")
    st.divider()
status_sidebar()
navigation.run()
