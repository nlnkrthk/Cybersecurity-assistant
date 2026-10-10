"""
Assistant chat: live agent trace, answer with guardrail notices, judge verdict,
evidence, and analyst feedback (thumbs, comment, attack-type correction).
"""
from typing import Any, Dict, Optional

import streamlit as st

import formatting as fmt
from api_client import ApiError
from components import answer_extras, attack_type_names, get_client

EXAMPLES = [
    "Investigate incident INC-000035 and tell me how to respond",
    "Has a similar SSH brute force incident occurred before?",
    "What traffic characteristics are typical of DDoS?",
    "Which of INC-000001, INC-000005, INC-002545 and INC-000035 should I investigate first?",
]
NO_CORRECTION = "— no correction —"

ss = st.session_state
ss.setdefault("thread_id", None)
ss.setdefault("chat", [])            # [{"role", "content", "result"?, "error"?}]
ss.setdefault("threads", {})         # thread_id -> title, conversations of this browser session
ss.setdefault("feedback_status", {})  # turn_id -> (kind, message)
ss.setdefault("pending_prompt", None)
ss.setdefault("thread_id_for_turn", {})  # turn_id -> thread_id, for feedback callbacks


# ============================================================================
# Conversation controls
# ============================================================================
def new_conversation() -> None:
    ss.thread_id, ss.chat = None, []


def load_conversation(thread_id: str) -> None:
    try:
        history = get_client().history(thread_id)
    except ApiError as exc:
        st.toast(f"Could not load the conversation: {exc.message}", icon="⚠️")
        return
    ss.thread_id = thread_id
    ss.chat = [{"role": m["role"], "content": m["content"]} for m in history["messages"]]


with st.sidebar:
    st.caption("CONVERSATION")
    st.button("New conversation", icon="➕", on_click=new_conversation, width="stretch")
    judge_on = st.toggle("Grade answers with LLM-as-Judge", key="judge_on",
                         help="One extra LLM call per answer: faithfulness, correctness, relevance, "
                              "actionability and safety, 1–5.")
    live_trace = st.toggle("Show agent steps live", value=True, key="live_trace")
    if ss.threads:
        options = list(ss.threads)
        current = options.index(ss.thread_id) if ss.thread_id in options else None
        picked = st.selectbox("Earlier conversations", options, index=current,
                              format_func=lambda t: ss.threads[t], placeholder="Pick a conversation")
        if picked and picked != ss.thread_id:
            load_conversation(picked)
            st.rerun()
    if ss.thread_id:
        st.caption(f"Thread `{ss.thread_id[:12]}…`")


# ============================================================================
# Feedback
# ============================================================================
def send_feedback(turn_id: str, rating: str, comment: Optional[str] = None,
                  corrected_label: Optional[str] = None) -> None:
    try:
        row = get_client().send_feedback(ss.thread_id_for_turn[turn_id], rating, turn_id=turn_id,
                                         comment=comment, corrected_label=corrected_label)
    except ApiError as exc:
        ss.feedback_status[turn_id] = ("error", exc.message)
        return
    note = f" Correction saved: **{row['corrected_label']}**." if row.get("corrected_label") else ""
    ss.feedback_status[turn_id] = ("ok", f"Thanks — feedback saved ({'👍' if rating == 'up' else '👎'}).{note}")


def on_thumbs(turn_id: str) -> None:
    value = ss.get(f"thumbs_{turn_id}")
    if value is not None:
        send_feedback(turn_id, "up" if value == 1 else "down")


def feedback_widget(result: Dict[str, Any]) -> None:
    turn_id = result.get("turn_id")
    if not turn_id or result.get("route") == "blocked":
        return
    ss.thread_id_for_turn[turn_id] = result["thread_id"]
    left, right = st.columns([1, 5], vertical_alignment="center")
    with left:
        st.feedback("thumbs", key=f"thumbs_{turn_id}", on_change=on_thumbs, args=(turn_id,))
    with right:
        with st.popover("Comment or correct", icon="✏️"):
            with st.form(f"feedback_form_{turn_id}", border=False):
                rating = st.segmented_control("Rating", ["up", "down"], default="down", required=True,
                                              format_func=lambda r: "👍 Helpful" if r == "up" else "👎 Not helpful")
                comment = st.text_area("Comment", max_chars=2000, placeholder="What was good or wrong?")
                label = None
                if fmt.has_flow_analysis(result):
                    predicted = fmt.predicted_label(result)
                    label = st.selectbox(
                        "Correct attack type", [NO_CORRECTION] + attack_type_names(),
                        help=f"The model predicted {predicted}." if predicted else None,
                    )
                if st.form_submit_button("Send feedback", type="primary"):
                    send_feedback(turn_id, rating, comment.strip() or None,
                                  None if label in (None, NO_CORRECTION) else label)
                    st.rerun()
    status = ss.feedback_status.get(turn_id)
    if status:
        (st.success if status[0] == "ok" else st.error)(status[1])


# ============================================================================
# Messages
# ============================================================================
def render_message(message: Dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["error"], icon="⚠️")
            return
        st.markdown(message["content"])
        result = message.get("result")
        if result:
            answer_extras(result)
            feedback_widget(result)


def run_turn(prompt: str) -> None:
    """Stream one turn, then store it in the chat history."""
    ss.chat.append({"role": "user", "content": prompt})
    render_message(ss.chat[-1])
    result, error, events = None, None, 0
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=live_trace)
        try:
            for kind, data in get_client().chat_stream(prompt, ss.thread_id, judge=judge_on):
                if kind == "trace":
                    events += 1
                    line = fmt.trace_line(data)
                    if line:
                        status.markdown(line)
                    label = fmt.status_label(data)
                    if label:
                        status.update(label=label, expanded=live_trace)
                elif kind == "result":
                    result = data
        except ApiError as exc:
            if exc.status is None and events == 0:  # stream not reachable: try the plain endpoint once
                try:
                    status.update(label="Streaming unavailable, waiting for the full answer…")
                    result = get_client().chat(prompt, ss.thread_id, judge=judge_on)
                except ApiError as retry_exc:
                    error = retry_exc.message
            else:
                error = exc.message
        if result:
            status.update(label=f"Done in {result.get('elapsed_ms', 0) / 1000:.1f}s", state="complete",
                          expanded=False)
        else:
            status.update(label="Failed", state="error", expanded=False)
            error = error or "The assistant returned no answer."

    if result:
        ss.thread_id = result["thread_id"]
        ss.threads.setdefault(ss.thread_id, prompt[:60] + ("…" if len(prompt) > 60 else ""))
        ss.chat.append({"role": "assistant", "content": result["answer"], "result": result})
    else:
        ss.chat.append({"role": "assistant", "content": "", "error": error})
    st.rerun()


st.title("💬 Incident response assistant")
if not ss.chat:
    st.markdown("Ask about an incident, a flow, past attacks or how to respond. "
                "The supervisor routes your question to the classification, risk & priority, "
                "incident search and response agents.")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"example_{i}", width="stretch"):
            ss.pending_prompt = example
            st.rerun()

for message in ss.chat:
    render_message(message)

prompt = st.chat_input("Ask about an incident, a flow or an attack type…", max_chars=4000)
if ss.pending_prompt and not prompt:
    prompt, ss.pending_prompt = ss.pending_prompt, None
if prompt:
    run_turn(prompt)
