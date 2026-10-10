"""
Answer quality: grade answers with LLM-as-Judge, and the analyst feedback dashboard
(approval, judge vs analyst agreement, corrections, export for retraining).
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import formatting as fmt
from components import call, get_client, judgement_card

ss = st.session_state

st.title("⚖️ Answer quality")
judge_tab, feedback_tab = st.tabs(["LLM-as-Judge", "Analyst feedback"])

with judge_tab:
    st.caption("Scores an answer 1–5 on faithfulness, correctness, relevance, actionability and safety, with "
               "rule-based caps (unverified IDs, unmatched figures, dangerous commands, missing caveats).")
    sources = ["Latest answer in my conversation", "Any question and answer"]
    source = st.segmented_control("Grade", sources, default=sources[0] if ss.get("thread_id") else sources[1],
                                  required=True, key="judge_source")
    if source == sources[0]:
        if not ss.get("thread_id"):
            st.info("Start a conversation on the Assistant page first.")
        else:
            last = next((m for m in reversed(ss.get("chat", [])) if m.get("result")), None)
            if last:
                st.markdown(f"> {last['content'][:400]}{'…' if len(last['content']) > 400 else ''}")
            if st.button("Grade with its evidence", type="primary"):
                with st.spinner("Judging…"):
                    ss.judge_result = call(lambda: get_client().judge(thread_id=ss.thread_id), "Judging")
    else:
        with st.form("judge_form"):
            question = st.text_area("Question", placeholder="What attack is INC-004393?")
            answer = st.text_area("Answer", height=160)
            route = st.selectbox("Route", list(fmt.ROUTE_LABELS)[:-1], index=None, placeholder="optional",
                                 format_func=lambda r: fmt.ROUTE_LABELS[r],
                                 help="Actionability is scored only for response and investigation answers.")
            if st.form_submit_button("Grade", type="primary"):
                if not question.strip() or not answer.strip():
                    st.error("Enter both a question and an answer.")
                else:
                    with st.spinner("Judging…"):
                        ss.judge_result = call(lambda: get_client().judge(question=question, answer=answer,
                                                                          route=route), "Judging")
        st.caption("Without evidence the judge can only check the answer against itself; grading a real "
                   "conversation turn uses its tool results.")
    if ss.get("judge_result"):
        judgement_card(ss.judge_result)
        st.caption(f"Judge model: `{ss.judge_result.get('judge_model')}`")

with feedback_tab:
    if st.button("Refresh", icon="🔄", type="tertiary"):
        st.rerun()
    stats = call(lambda: get_client().feedback_stats(), "Loading feedback")
    if stats is not None:
        if not stats.get("total"):
            st.info("No feedback yet. Rate answers on the Assistant page (👍/👎, comment, attack-type correction).")
        else:
            agreement = stats["judge_agreement"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ratings", stats["total"], f"{stats['up']} 👍 / {stats['down']} 👎", delta_color="off", delta_arrow="off", border=True)
            c2.metric("Approval rate", fmt.pct(stats["approval_rate"]), border=True)
            c3.metric("Judge vs analyst agreement", fmt.pct(agreement.get("agreement_rate")),
                      f"{agreement['n']} judged answers", delta_color="off", delta_arrow="off", border=True,
                      help="Judge 'pass' counts as agreeing with 👍, 'needs review' or 'fail' with 👎.")
            c4.metric("Label corrections", stats["corrections"]["total"], border=True)

            left, right = st.columns(2)
            with left:
                st.markdown("**Approval by route**")
                routes = stats.get("by_route") or {}
                fig = go.Figure(go.Bar(
                    x=[fmt.ROUTE_LABELS.get(r, r) for r in routes], y=[v["approval_rate"] for v in routes.values()],
                    text=[f"{v['up']}/{v['n']}" for v in routes.values()], textposition="auto",
                    marker_color="#2ca02c",
                ))
                fig.update_layout(height=280, yaxis=dict(range=[0, 1], tickformat=".0%"),
                                  margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, key="approval_chart")
            with right:
                st.markdown("**Judge verdict vs analyst rating**")
                st.dataframe(pd.DataFrame(
                    {"👍 analyst": [agreement["pass_up"], agreement["not_pass_up"]],
                     "👎 analyst": [agreement["pass_down"], agreement["not_pass_down"]]},
                    index=["judge pass", "judge needs review / fail"],
                ))
                pairs = stats["corrections"]["top_pairs"]
                if pairs:
                    st.markdown("**Most common corrections**")
                    st.dataframe(pd.DataFrame(pairs), hide_index=True)

            comments = stats.get("recent_comments") or []
            if comments:
                st.markdown("**Recent comments**")
                for c in comments:
                    st.markdown(f"{'👍' if c['rating'] == 'up' else '👎'} *{c['comment']}* — "
                                f"{fmt.ROUTE_LABELS.get(c['route'], c['route'])}, {str(c['created_at'])[:16].replace('T', ' ')} UTC")

            st.markdown("**All feedback**")
            f1, f2, f3 = st.columns(3)
            rating = f1.selectbox("Rating", ["up", "down"], index=None, placeholder="Any")
            route = f2.selectbox("Route", list(fmt.ROUTE_LABELS), index=None, placeholder="Any",
                                 format_func=lambda r: fmt.ROUTE_LABELS[r])
            corrected_only = f3.checkbox("Corrections only")
            rows = call(lambda: get_client().feedback_list(rating=rating, route=route,
                                                           corrected_only=corrected_only, limit=200),
                        "Loading feedback")
            if rows:
                columns = ["created_at", "rating", "route", "question", "incident_id", "predicted_label", "corrected_label",
                           "judge_verdict", "comment", "llm_model"]
                frame = pd.DataFrame(rows)
                st.dataframe(frame[[c for c in columns if c in frame.columns]], hide_index=True)

            st.markdown("**Retraining data**")
            st.caption("Corrected flows with their 24 features (majority label per flow). Retrain with "
                       "`scripts/feedback/retrain_classifier.py`; a new classifier is activated only if its "
                       "macro F1 does not drop.")
            if stats["corrections"]["total"]:
                csv = call(lambda: get_client().feedback_export_csv(), "Export")
                if csv:
                    st.download_button("Download corrections CSV", csv, "feedback_corrections.csv", "text/csv",
                                       icon="⬇️")
