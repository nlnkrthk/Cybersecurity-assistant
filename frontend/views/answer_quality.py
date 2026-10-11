"""
Quality & Feedback: grade answers with the AI quality checker (LLM-as-Judge), and the analyst
feedback dashboard (approval, checker vs analyst agreement, corrections, export for retraining).
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import formatting as fmt
from components import call, get_client, judgement_card, page_header

ss = st.session_state
GRADABLE_ROUTES = ["classify", "risk", "search", "recommend", "investigate", "general"]

page_header("Quality & Feedback",
            "Check how reliable the assistant's answers are, and review what analysts said about them.",
            how_to=["**Grade an answer:** have the AI quality checker score the latest answer of your conversation "
                    "(using the data the agents found), or any question and answer you paste in.",
                    "Scores run from 1 to 5 for faithfulness, correctness, relevance, actionability and safety.",
                    "**Analyst feedback:** see approval rates, how often the quality checker agrees with analysts, "
                    "and which attack types analysts corrected.",
                    "Corrections can be downloaded and are used to retrain the attack classifier."],
            icon=":material/fact_check:")

judge_tab, feedback_tab = st.tabs(["Grade an answer", "Analyst feedback"])

with judge_tab:
    sources = ["Latest answer in my conversation", "Paste a question and answer"]
    source = st.segmented_control("What to grade", sources, default=sources[0] if ss.get("thread_id") else sources[1],
                                  required=True, key="judge_source")
    if source == sources[0]:
        if not ss.get("thread_id"):
            st.info("Ask the AI Assistant a question first; its latest answer can then be graded here.",
                    icon=":material/info:")
        else:
            last = next((m for m in reversed(ss.get("chat", [])) if m.get("result")), None)
            if last:
                with st.container(border=True):
                    st.caption("Latest answer")
                    st.markdown(f"{last['content'][:600]}{'…' if len(last['content']) > 600 else ''}")
            if st.button("Grade this answer", type="primary", icon=":material/grading:"):
                with st.spinner("Grading…"):
                    ss.judge_result = call(lambda: get_client().judge(thread_id=ss.thread_id), "Grading")
    else:
        with st.form("judge_form"):
            question = st.text_area("Question", placeholder="e.g. What attack is INC-004393?")
            answer = st.text_area("Answer to grade", height=160)
            route = st.selectbox("Type of question (optional)", GRADABLE_ROUTES, index=None, placeholder="Not specified",
                                 format_func=lambda r: fmt.ROUTE_LABELS[r],
                                 help="Actionability is only scored for response advice and full investigations.")
            if st.form_submit_button("Grade", type="primary", icon=":material/grading:"):
                if not question.strip() or not answer.strip():
                    st.error("Enter both a question and an answer.")
                else:
                    with st.spinner("Grading…"):
                        ss.judge_result = call(lambda: get_client().judge(question=question, answer=answer,
                                                                          route=route), "Grading")
        st.caption("A pasted answer has no data behind it, so the checker can only judge it on its own; grading "
                   "a real conversation answer also checks it against the data the agents used.")
    if ss.get("judge_result"):
        judgement_card(ss.judge_result)
        st.caption(f"Graded by `{ss.judge_result.get('judge_model')}`")

with feedback_tab:
    st.button("Refresh", icon=":material/refresh:", key="refresh_feedback")
    stats = call(lambda: get_client().feedback_stats(), "Loading feedback")
    if stats is not None:
        if not stats.get("total"):
            st.info("No feedback yet. Rate answers in the AI Assistant with 👍 / 👎, add comments, or correct an "
                    "attack type; it will show up here.", icon=":material/info:")
        else:
            agreement = stats["judge_agreement"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ratings", stats["total"], f"{stats['up']} 👍 · {stats['down']} 👎", delta_color="off",
                      delta_arrow="off", border=True)
            c2.metric("Approval rate", fmt.pct(stats["approval_rate"]), border=True,
                      help="Share of rated answers that analysts found helpful.")
            c3.metric("Checker agrees with analysts", fmt.pct(agreement.get("agreement_rate")),
                      f"on {agreement['n']} graded answers", delta_color="off", delta_arrow="off", border=True,
                      help="A 'pass' from the quality checker counts as agreeing with 👍; 'needs review' or "
                           "'fail' with 👎.")
            c4.metric("Attack-type corrections", stats["corrections"]["total"], border=True,
                      help="Used to retrain the attack classifier.")

            left, right = st.columns(2)
            with left:
                st.markdown("**Approval by type of question**")
                routes = stats.get("by_route") or {}
                fig = go.Figure(go.Bar(
                    x=[fmt.ROUTE_LABELS.get(r, r) for r in routes], y=[v["approval_rate"] for v in routes.values()],
                    text=[f"{v['up']} of {v['n']}" for v in routes.values()], textposition="auto",
                    marker_color="#16A34A",
                ))
                fig.update_layout(height=280, yaxis=dict(range=[0, 1], tickformat=".0%"), plot_bgcolor="white",
                                  margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, key="approval_chart")
            with right:
                st.markdown("**Quality checker vs analyst rating**")
                st.dataframe(pd.DataFrame(
                    {"Analyst 👍": [agreement["pass_up"], agreement["not_pass_up"]],
                     "Analyst 👎": [agreement["pass_down"], agreement["not_pass_down"]]},
                    index=["Checker: pass", "Checker: needs review / fail"],
                ))
                pairs = stats["corrections"]["top_pairs"]
                if pairs:
                    st.markdown("**Most common corrections**")
                    st.dataframe(pd.DataFrame(pairs), hide_index=True, column_config={
                        "predicted": "Model said", "corrected": "Analyst corrected to", "count": "Times"})

            comments = stats.get("recent_comments") or []
            if comments:
                st.markdown("**Recent comments**")
                for c in comments:
                    st.markdown(f"{'👍' if c['rating'] == 'up' else '👎'} *{c['comment']}* — "
                                f"{fmt.ROUTE_LABELS.get(c['route'], c['route'])}, "
                                f"{str(c['created_at'])[:16].replace('T', ' ')} UTC")

            with st.expander("All feedback", icon=":material/table_rows:"):
                f1, f2, f3 = st.columns(3)
                rating = f1.selectbox("Rating", ["up", "down"], index=None, placeholder="Any",
                                      format_func=lambda r: "👍 Helpful" if r == "up" else "👎 Not helpful")
                route = f2.selectbox("Type of question", list(fmt.ROUTE_LABELS), index=None, placeholder="Any",
                                     format_func=lambda r: fmt.ROUTE_LABELS[r])
                corrected_only = f3.checkbox("Only answers with a correction")
                rows = call(lambda: get_client().feedback_list(rating=rating, route=route,
                                                               corrected_only=corrected_only, limit=200),
                            "Loading feedback")
                if rows:
                    columns = {"created_at": "Time (UTC)", "rating": "Rating", "route": "Type", "question": "Question",
                               "incident_id": "Incident", "predicted_label": "Model said",
                               "corrected_label": "Corrected to", "judge_verdict": "Checker", "comment": "Comment"}
                    frame = pd.DataFrame(rows)
                    frame = frame[[c for c in columns if c in frame.columns]].rename(columns=columns)
                    st.dataframe(frame, hide_index=True)
                else:
                    st.caption("No feedback matches these filters.")

            with st.container(border=True):
                st.markdown("**Retraining data**")
                st.caption("Corrected flows with their 24 metrics (the majority label when analysts disagree). "
                           "Retrain with `scripts/feedback/retrain_classifier.py`; a new classifier is only "
                           "activated if it is not less accurate than the current one.")
                if stats["corrections"]["total"]:
                    csv = call(lambda: get_client().feedback_export_csv(), "Export")
                    if csv:
                        st.download_button("Download corrections (CSV)", csv, "feedback_corrections.csv",
                                           "text/csv", icon=":material/download:")
                else:
                    st.caption("No corrections yet.")
