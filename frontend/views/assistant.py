"""
AI Assistant: chat with a live step-by-step progress tracker, answers with safety notices,
quality check, process and data used, and analyst feedback (thumbs, comment, correction).
"""
import time
from typing import Any, Dict, List, Optional

import streamlit as st

import formatting as fmt
from api_client import ApiError
from components import answer_extras, attack_type_names, get_client, page_header, show_pipeline

EXAMPLES = [
    ("Investigate an incident", "Investigate incident INC-000035 and tell me how to respond"),
    ("Find similar past attacks", "Has a similar SSH brute force incident occurred before?"),
    ("Learn an attack's traffic pattern", "What traffic characteristics are typical of DDoS?"),
    ("Decide what to look at first", "Which of INC-000001, INC-000005, INC-002545 and INC-000035 should I investigate first?"),
]
NO_CORRECTION = "No correction"

ss = st.session_state
ss.setdefault("thread_id", None)
ss.setdefault("chat", [])            # [{"role", "content", "result"?, "error"?}]
ss.setdefault("threads", {})         # thread_id -> title, conversations of this browser session
ss.setdefault("feedback_status", {})  # turn_id -> (kind, message)
ss.setdefault("pending_prompt", None)
ss.setdefault("thread_id_for_turn", {})  # turn_id -> thread_id, for feedback callbacks


# ============================================================================
# Conversation controls (sidebar)
# ============================================================================
def new_conversation() -> None:
    ss.thread_id, ss.chat = None, []


def load_conversation(thread_id: str) -> None:
    try:
        history = get_client().history(thread_id)
    except ApiError as exc:
        st.toast(f"Could not load the conversation: {exc.message}", icon=":material/error:")
        return
    ss.thread_id = thread_id
    ss.chat = [{"role": m["role"], "content": m["content"]} for m in history["messages"]]


with st.sidebar:
    st.divider()
    st.markdown('<div class="side-title">Chat settings</div>', unsafe_allow_html=True)
    st.button("New conversation", icon=":material/add_comment:", on_click=new_conversation,
              type="primary", width="stretch")
    judge_on = st.toggle("Quality-check each answer", key="judge_on",
                         help="An independent AI grader scores every answer from 1 to 5 (faithfulness, "
                              "correctness, relevance, actionability, safety). Adds a few seconds per answer.")
    if ss.threads:
        options = list(ss.threads)
        current = options.index(ss.thread_id) if ss.thread_id in options else None
        picked = st.selectbox("Earlier conversations", options, index=current,
                              format_func=lambda t: ss.threads[t], placeholder="Open an earlier conversation")
        if picked and picked != ss.thread_id:
            load_conversation(picked)
            st.rerun()


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
    ss.feedback_status[turn_id] = ("ok", f"Thanks, your feedback was saved.{note}")


def on_thumbs(turn_id: str) -> None:
    value = ss.get(f"thumbs_{turn_id}")
    if value is not None:
        send_feedback(turn_id, "up" if value == 1 else "down")


def feedback_widget(result: Dict[str, Any]) -> None:
    turn_id = result.get("turn_id")
    if not turn_id or result.get("route") == "blocked":
        return
    ss.thread_id_for_turn[turn_id] = result["thread_id"]
    label, thumbs, more = st.columns([0.9, 0.8, 4], vertical_alignment="center")
    label.caption("Was this helpful?")
    with thumbs:
        st.feedback("thumbs", key=f"thumbs_{turn_id}", on_change=on_thumbs, args=(turn_id,))
    with more:
        flow = fmt.has_flow_analysis(result)
        with st.popover("Add a comment" + (" or correct the attack type" if flow else ""), icon=":material/edit:"):
            with st.form(f"feedback_form_{turn_id}", border=False):
                rating = st.segmented_control("Your rating", ["up", "down"], default="down", required=True,
                                              format_func=lambda r: "👍 Helpful" if r == "up" else "👎 Not helpful")
                comment = st.text_area("Comment (optional)", max_chars=2000, placeholder="What was good or wrong?")
                label_choice = None
                if flow:
                    predicted = fmt.predicted_label(result)
                    label_choice = st.selectbox(
                        "Correct attack type (optional)", [NO_CORRECTION] + attack_type_names(),
                        help=(f"The model predicted {predicted}. " if predicted else "")
                        + "Corrections appear in later answers and are used to retrain the classifier.",
                    )
                if st.form_submit_button("Send feedback", type="primary", icon=":material/send:"):
                    send_feedback(turn_id, rating, comment.strip() or None,
                                  None if label_choice in (None, NO_CORRECTION) else label_choice)
                    st.rerun()
    status = ss.feedback_status.get(turn_id)
    if status:
        (st.success if status[0] == "ok" else st.error)(status[1], icon=":material/check_circle:"
                                                         if status[0] == "ok" else ":material/error:")


# ============================================================================
# Messages
# ============================================================================
def render_message(message: Dict[str, Any]) -> None:
    with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user"
                         else ":material/smart_toy:"):
        if message.get("error"):
            st.error(message["error"], icon=":material/error:")
            return
        st.markdown(message["content"])
        result = message.get("result")
        if result:
            answer_extras(result)
            feedback_widget(result)


def run_turn(prompt: str) -> None:
    """Stream one turn with the live progress tracker, then store it in the chat history."""
    ss.chat.append({"role": "user", "content": prompt})
    render_message(ss.chat[-1])
    result, error = None, None
    events: List[Dict[str, Any]] = []
    started = time.perf_counter()
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        tracker = st.empty()
        show_pipeline(tracker, fmt.pipeline_steps(events, judge=judge_on, running=True), "Working on your question…")
        try:
            for kind, data in get_client().chat_stream(prompt, ss.thread_id, judge=judge_on):
                if kind == "trace":
                    events.append(data)
                    show_pipeline(tracker, fmt.pipeline_steps(events, judge=judge_on, running=True),
                                  "Working on your question…", f"{time.perf_counter() - started:.0f}s")
                elif kind == "result":
                    result = data
        except ApiError as exc:
            if exc.status is None and not events:  # stream not reachable: try the plain endpoint once
                try:
                    tracker.info("Live progress is unavailable; waiting for the full answer…", icon=":material/hourglass:")
                    result = get_client().chat(prompt, ss.thread_id, judge=judge_on)
                except ApiError as retry_exc:
                    error = retry_exc.message
            else:
                error = exc.message
        if not result:
            error = error or "The assistant returned no answer."
            show_pipeline(tracker, fmt.pipeline_steps(events, judge=judge_on), "Stopped")

    if result:
        ss.thread_id = result["thread_id"]
        ss.threads.setdefault(ss.thread_id, prompt[:60] + ("…" if len(prompt) > 60 else ""))
        ss.chat.append({"role": "assistant", "content": result["answer"], "result": result})
    else:
        ss.chat.append({"role": "assistant", "content": "", "error": error})
    st.rerun()


page_header("AI Assistant",
            "Ask about an incident, a network flow, past attacks or how to respond. Specialist agents gather "
            "the facts; you can follow each step while they work.",
            how_to=["Type a question below, or pick one of the examples.",
                    "Watch the progress panel: each agent and the data it looks up are shown as they run.",
                    "Read the answer. Under it you can open **How this answer was produced** and **Data used**.",
                    "If details such as byte counts are missing, the assistant asks for them; just reply.",
                    "Rate the answer with 👍 / 👎. If an attack type is wrong, correct it; this improves the model.",
                    "Use **Quality-check each answer** in the sidebar to have every answer graded."],
            icon=":material/smart_toy:")

if not ss.chat:
    st.markdown("##### Try an example")
    cols = st.columns(2)
    for i, (title, example) in enumerate(EXAMPLES):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{title}**")
            st.caption(example)
            if st.button("Ask this", key=f"example_{i}", icon=":material/send:"):
                ss.pending_prompt = example
                st.rerun()

for message in ss.chat:
    render_message(message)

prompt = st.chat_input("Ask about an incident (e.g. INC-000035), a flow, or an attack type…", max_chars=4000)
if ss.pending_prompt and not prompt:
    prompt, ss.pending_prompt = ss.pending_prompt, None
if prompt:
    run_turn(prompt)
