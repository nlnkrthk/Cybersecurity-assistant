"""
Home: what each tool does and when to use it.
"""
import streamlit as st

from components import page_header, priority_legend

TOOLS = [
    {
        "page": "views/assistant.py", "icon": ":material/smart_toy:", "title": "AI Assistant",
        "what": "Ask questions in plain English. Specialist AI agents look up incidents, classify attacks, "
                "score risk, search past cases and recommend response steps.",
        "when": "You want an explanation or a response plan, e.g. “Investigate INC-000035 and tell me how to respond”.",
    },
    {
        "page": "views/flow_analyzer.py", "icon": ":material/query_stats:", "title": "Flow Analysis",
        "what": "Check one network flow with the machine-learning models: attack type, intrusion risk, "
                "how unusual it is, priority, and which metrics drove the result.",
        "when": "You have a suspicious flow (or incident ID) and want a fast, data-only verdict.",
    },
    {
        "page": "views/incident_search.py", "icon": ":material/manage_search:", "title": "Incident Search",
        "what": "Search about 97,000 historical incidents by description, attack type or port, and see what "
                "each attack type typically looks like.",
        "when": "You want to know whether something similar has happened before.",
    },
    {
        "page": "views/triage_queue.py", "icon": ":material/format_list_numbered:", "title": "Triage Queue",
        "what": "Rank many flows or incidents by urgency from a CSV file or a list of incident IDs.",
        "when": "You have a batch of alerts and need to decide what to look at first.",
    },
    {
        "page": "views/answer_quality.py", "icon": ":material/fact_check:", "title": "Quality & Feedback",
        "what": "Grade any answer with the AI quality checker and see how analysts rated the assistant, "
                "including corrected attack types used to retrain the model.",
        "when": "You want to check the assistant's reliability or review analyst feedback.",
    },
]

page_header("Incident Response Assistant",
            "Investigate, classify and prioritise network security incidents with AI agents and "
            "machine-learning models trained on the CIC-IDS2017 dataset.", icon=":material/shield:")

GOOD_TO_KNOW = ("- **Priorities** run from P1 (act now) to P4 (monitor).\n"
                "- Results based on **partial input** are marked as less reliable.\n"
                "- Every AI answer passes a **safety check** before and an **answer check** after.\n"
                "- Your 👍 / 👎 and corrections **improve the models** over time.")

cards = TOOLS + [None]  # the last slot is the "Good to know" card
for row_start in range(0, len(cards), 3):  # one row of columns at a time keeps the order on phones
    for col, tool in zip(st.columns(3), cards[row_start:row_start + 3]):
        with col.container(border=True, height="stretch"):
            if tool is None:
                st.markdown("#### :material/info: Good to know")
                st.markdown(GOOD_TO_KNOW)
                continue
            i = TOOLS.index(tool)
            st.markdown(f"#### {tool['icon']} {tool['title']}")
            st.markdown(tool["what"])
            st.caption(f"**Use it when:** {tool['when']}")
            if st.button(f"Open {tool['title']}", key=f"open_{i}", type="primary" if i == 0 else "secondary",
                         icon=":material/arrow_forward:", icon_position="right", width="stretch"):
                st.switch_page(tool["page"])

st.markdown("##### Priority levels")
priority_legend()

with st.expander("How the AI assistant works", icon=":material/account_tree:"):
    st.markdown(
        "1. **Safety check** – the question is screened for unsafe requests; secrets such as passwords are removed.\n"
        "2. **Planning** – a supervisor decides which specialist agents are needed.\n"
        "3. **Specialist agents** – Classification, Risk & Priority, Incident Search and Response agents call "
        "the machine-learning models and the incident database.\n"
        "4. **Answer** – the findings are combined into one reply, then checked for invented incident IDs, "
        "unsupported figures, wrong priorities and dangerous commands.\n"
        "5. **Optional quality grading** – an independent AI grader scores the answer from 1 to 5.")
