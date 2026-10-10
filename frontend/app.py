"""
Streamlit front end of the Cybersecurity Incident Response Assistant.

    streamlit run frontend/app.py          (or scripts/frontend/run_frontend.py)

A thin client: every page calls the FastAPI backend (API_BASE_URL, default
http://localhost:8000) through api_client.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from components import status_sidebar  # noqa: E402

st.set_page_config(page_title="Incident Response Assistant", page_icon="🛡️", layout="wide")

VIEWS = Path(__file__).resolve().parent / "views"
pages = {
    "Investigate": [
        st.Page(VIEWS / "assistant.py", title="Assistant", icon="💬", default=True),
        st.Page(VIEWS / "flow_analyzer.py", title="Flow analyzer", icon="🔬"),
        st.Page(VIEWS / "incident_search.py", title="Incident search", icon="🔎"),
        st.Page(VIEWS / "triage_queue.py", title="Triage queue", icon="🚨"),
    ],
    "Quality": [
        st.Page(VIEWS / "answer_quality.py", title="Answer quality", icon="⚖️"),
    ],
}

navigation = st.navigation(pages)
st.sidebar.markdown("### 🛡️ Incident Response Assistant")
st.sidebar.caption("CIC-IDS2017 · multi-agent analysis")
status_sidebar()
navigation.run()
