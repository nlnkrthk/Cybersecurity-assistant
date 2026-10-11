"""
Graph tests with a scripted fake LLM (no API calls). The tools and ML models are real.
"""
import threading
from collections import deque

import pytest
from langchain_core.messages import AIMessage

from agents.graph import OUT_OF_SCOPE_REPLY, Assistant, RouteDecision, next_step
from agents.prompts import AGENT_PROMPTS, DIRECT_REPLY_ENDING, FINDINGS_ENDING, ROUTER_PROMPT
from agents.specialists import MAX_TOOL_ROUNDS, build_task
from agents.state import next_stage, plan_stages


def agent_of(messages):
    """Which agent a request comes from, by its system prompt."""
    return next(name for name, prompt in AGENT_PROMPTS.items() if messages[0].content.startswith(prompt))


class FakeLLM:
    """Scripted chat model: separate queues for the router, tool-using agents and the final answer.

    agent_replies is one queue in call order across all agents, or a dict of queues per agent
    (needed when agents run in parallel, so the call order between agents is not fixed).
    """

    def __init__(self, routes=(), agent_replies=(), final_replies=()):
        self.routes = deque(routes)                # RouteDecision | "refusal" | "garbage" | Exception
        if isinstance(agent_replies, dict):        # AIMessage | callable(messages) -> AIMessage
            self.agent_replies = {agent: deque(replies) for agent, replies in agent_replies.items()}
        else:
            self.agent_replies = deque(agent_replies)
        self.final_replies = deque(final_replies)  # str
        self.router_inputs, self.agent_inputs, self.final_inputs = [], [], []
        self.inputs_by_agent = {}

    # router fallback: llm.with_structured_output(RouteDecision, method=..., include_raw=True)
    def with_structured_output(self, schema, **kwargs):
        fake = self

        class Router:
            def invoke(self, messages):
                fake.router_inputs.append(messages)
                item = fake.routes.popleft()
                if isinstance(item, Exception):
                    raise item
                if item == "refusal":
                    return {"raw": AIMessage("", response_metadata={"stop_reason": "refusal"}), "parsed": None,
                            "parsing_error": ValueError("empty")}
                if item == "garbage":
                    return {"raw": AIMessage("not json"), "parsed": None, "parsing_error": ValueError("bad json")}
                return {"raw": AIMessage(""), "parsed": item, "parsing_error": None}

        return Router()

    def _route_reply(self, messages):
        """Router's first attempt: a plain request that must return route JSON."""
        self.router_inputs.append(messages)
        item = self.routes.popleft()
        if isinstance(item, Exception):
            raise item
        if item == "refusal":
            return AIMessage("", response_metadata={"stop_reason": "refusal"})
        if item == "garbage":
            return AIMessage("not json")  # the router then tries structured output (next queue item)
        return AIMessage(item.model_dump_json())

    # specialists: llm.bind_tools(tools).invoke(messages)
    def bind_tools(self, tools):
        fake = self

        class Bound:
            def invoke(self, messages):
                agent = agent_of(messages)
                fake.agent_inputs.append(list(messages))
                fake.inputs_by_agent.setdefault(agent, []).append(list(messages))
                queue = fake.agent_replies[agent] if isinstance(fake.agent_replies, dict) else fake.agent_replies
                reply = queue.popleft()
                return reply(messages) if callable(reply) else reply

        return Bound()

    # router (plain JSON request) and final answer: llm.invoke(messages)
    def invoke(self, messages):
        if messages and messages[0].content.startswith(ROUTER_PROMPT):
            return self._route_reply(messages)
        self.final_inputs.append(messages)
        return AIMessage(self.final_replies.popleft())


def route(name):
    return RouteDecision(route=name, reason="test")


def tool_call(name, args, call_id="call_1"):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def text(content):
    return AIMessage(content)


# ---------------------------------------------------------------- single-agent route
def test_classify_route_runs_one_agent_with_real_tool():
    llm = FakeLLM(
        routes=[route("classify")],
        agent_replies=[tool_call("classify_attack_type", {"incident_id": "INC-004393"}), text("It is Infiltration.")],
    )
    result = Assistant(llm=llm).ask("What attack is INC-004393?")

    assert result["route"] == "classify"
    assert result["agents_run"] == ["classification"]
    assert result["answer"] == "It is Infiltration."  # the agent's reply, no final rewrite
    assert llm.final_inputs == []
    assert llm.agent_inputs[0][0].content.endswith(DIRECT_REPLY_ENDING)
    assert [e for e in result["trace"] if e["type"] == "final"][0]["direct"]
    assert result["evidence"][0]["result"]["attack_type"]["prediction"] == "Infiltration"
    assert result["flow_args"] == {"incident_id": "INC-004393"}


def test_single_agent_findings_go_through_final_answer_when_direct_reply_is_off():
    llm = FakeLLM(
        routes=[route("classify")],
        agent_replies=[tool_call("classify_attack_type", {"incident_id": "INC-004393"}), text("- Infiltration, 90%")],
        final_replies=["It is Infiltration."],
    )
    result = Assistant(llm=llm, direct_reply=False).ask("What attack is INC-004393?")
    assert result["answer"] == "It is Infiltration."
    assert llm.agent_inputs[0][0].content.endswith(FINDINGS_ENDING)
    assert "- Infiltration, 90%" in llm.final_inputs[0][1].text  # findings reach the final answer node


def test_empty_direct_reply_falls_back_to_the_final_answer_node():
    llm = FakeLLM(routes=[route("classify")], agent_replies=[text("")], final_replies=["rescued"])
    result = Assistant(llm=llm).ask("classify INC-000001")
    assert result["answer"] == "rescued"


# ---------------------------------------------------------------- multi-agent route
INVESTIGATE_REPLIES = {
    "classification": [text("class findings")], "risk_priority": [text("risk findings")],
    "incident_search": [text("search findings")], "response": [text("response plan")],
}


def test_investigate_runs_two_parallel_stages_and_shares_findings():
    llm = FakeLLM(routes=[route("investigate")], agent_replies=INVESTIGATE_REPLIES, final_replies=["full report"])
    result = Assistant(llm=llm).ask("Investigate INC-000035")

    assert result["agents_run"] == ["classification", "risk_priority", "incident_search", "response"]
    assert list(result["findings"]) == result["agents_run"]  # merged in plan order
    route_event = next(e for e in result["trace"] if e["type"] == "route")
    assert route_event["stages"] == [["classification", "risk_priority"], ["incident_search", "response"]]
    # stage 2 sees stage 1's findings; agents of the same stage do not see each other's
    for agent in ("incident_search", "response"):
        task = llm.inputs_by_agent[agent][0][1].text
        assert "class findings" in task and "risk findings" in task
    assert "class findings" not in llm.inputs_by_agent["risk_priority"][0][1].text
    assert "search findings" not in llm.inputs_by_agent["response"][0][1].text
    assert llm.inputs_by_agent["response"][0][0].content.endswith(FINDINGS_ENDING)
    final_prompt = llm.final_inputs[0][1].text
    for findings in ["class findings", "risk findings", "search findings", "response plan"]:
        assert findings in final_prompt
    assert result["answer"] == "full report"


def test_agents_of_a_stage_really_run_at_the_same_time():
    barrier = threading.Barrier(2, timeout=10)  # breaks (and the test fails) if the agents run one after another

    def meet(reply):
        def answer(messages):
            barrier.wait()
            return text(reply)
        return answer

    llm = FakeLLM(
        routes=[route("investigate")],
        agent_replies={"classification": [meet("class")], "risk_priority": [meet("risk")],
                       "incident_search": [text("search")], "response": [text("plan")]},
        final_replies=["report"],
    )
    result = Assistant(llm=llm).ask("Investigate INC-000035")
    assert result["findings"]["classification"] == "class" and result["findings"]["risk_priority"] == "risk"


def test_parallel_off_runs_agents_one_by_one():
    llm = FakeLLM(
        routes=[route("investigate")],
        agent_replies=[text("class findings"), text("risk findings"), text("search findings"), text("response plan")],
        final_replies=["full report"],
    )
    result = Assistant(llm=llm, parallel=False).ask("Investigate INC-000035")
    assert result["agents_run"] == ["classification", "risk_priority", "incident_search", "response"]
    assert "class findings" in llm.inputs_by_agent["risk_priority"][0][1].text
    assert "search findings" in llm.inputs_by_agent["response"][0][1].text


def test_follow_up_from_a_parallel_stage_stops_the_turn():
    llm = FakeLLM(
        routes=[route("investigate")],
        agent_replies={"classification": [text("QUESTION: Which flow?")], "risk_priority": [text("risk")]},
    )
    result = Assistant(llm=llm).ask("Investigate it")
    assert result["needs_follow_up"] and result["answer"] == "Which flow?"
    assert "incident_search" not in llm.inputs_by_agent and llm.final_inputs == []


def test_flow_identified_in_the_first_stage_is_passed_to_the_next_stage():
    llm = FakeLLM(
        routes=[route("investigate")],
        agent_replies={
            "classification": [tool_call("explain_flow", {"incident_id": "INC-000035"}), text("SSH-Patator")],
            "risk_priority": [text("risk")], "incident_search": [text("search")], "response": [text("plan")],
        },
        final_replies=["report"],
    )
    Assistant(llm=llm).ask("Investigate INC-000035")
    search_task = llm.inputs_by_agent["incident_search"][0][1].text
    assert "Flow already identified this turn" in search_task and "INC-000035" in search_task


# ---------------------------------------------------------------- follow-up across turns
def test_missing_metrics_ask_follow_up_then_resume():
    llm = FakeLLM(
        routes=[route("risk"), route("risk")],
        agent_replies=[
            tool_call("assess_intrusion_risk", {"flow_duration": "2 s", "total_fwd_packets": 10, "total_bwd_packets": 8}),
            text("Not suspicious."),
        ],
    )
    assistant = Assistant(llm=llm)
    thread = assistant.new_thread_id()

    first = assistant.ask("Is this suspicious? 2 s, 10 packets out, 8 back", thread)
    assert first["needs_follow_up"]
    assert "byte counts" in first["answer"]
    assert llm.final_inputs == []  # follow-up skips the final answer node

    second = assistant.ask("1.2 KB sent, 9.6 KB received", thread)
    assert not second["needs_follow_up"]
    assert second["answer"] == "Not suspicious."
    router_context = llm.router_inputs[1][1].text
    assert "asked for missing metrics" in router_context
    agent_task = llm.agent_inputs[1][1].text
    assert "previous assistant turn asked the user for missing metrics" in agent_task
    assert "Latest flow analysed earlier in this conversation" in agent_task


# ---------------------------------------------------------------- risk answers always carry a priority
FULL_FLOW = {"flow_duration": "2 s", "total_fwd_packets": 10, "total_bwd_packets": 8,
             "total_len_fwd": "1.2 KB", "total_len_bwd": "4 KB"}


def _priority_calls(result):
    return [e for e in result["evidence"] if e["tool"] == "prioritize_flow"]


def test_risk_agent_runs_prioritiser_when_llm_skips_it():
    llm = FakeLLM(
        routes=[route("risk")],
        agent_replies=[tool_call("assess_intrusion_risk", FULL_FLOW), text("- low risk"),
                       text("- low risk\n- priority P4")],
    )
    result = Assistant(llm=llm).ask("Is this flow suspicious?")

    auto = _priority_calls(result)
    assert len(auto) == 1 and auto[0]["auto"] and auto[0]["args"] == FULL_FLOW
    assert auto[0]["result"]["priority"]["priority"] in {"P1", "P2", "P3", "P4"}
    assert result["findings"]["risk_priority"] == "- low risk\n- priority P4"
    rewrite_request = llm.agent_inputs[-1][-1].text
    assert "prioritisation tool was run" in rewrite_request and auto[0]["result"]["priority"]["priority"] in rewrite_request
    assert any(e.get("auto") for e in result["trace"] if e["type"] == "tool_call")


def test_risk_agent_does_not_reprioritise_when_priority_exists():
    llm = FakeLLM(
        routes=[route("risk")],
        agent_replies=[tool_call("prioritize_flow", {"incident_id": "INC-000002"}), text("P4")],
    )
    result = Assistant(llm=llm).ask("Is incident INC-000002 dangerous?")
    assert len(_priority_calls(result)) == 1 and not _priority_calls(result)[0].get("auto")
    assert len(llm.agent_inputs) == 2


def test_auto_priority_is_appended_when_rewrite_fails():
    llm = FakeLLM(
        routes=[route("risk")],
        agent_replies=[tool_call("assess_intrusion_risk", {"incident_id": "INC-000002"}), text("- low risk"),
                       tool_call("list_attack_types", {})],  # rewrite ignored the instruction
    )
    result = Assistant(llm=llm).ask("Is incident INC-000002 dangerous?")
    priority = _priority_calls(result)[0]["result"]["priority"]["priority"]
    findings = result["findings"]["risk_priority"]
    assert findings.startswith("- low risk") and f"Priority (prioritize_flow): {priority}" in findings


def test_agent_question_line_becomes_follow_up():
    llm = FakeLLM(routes=[route("classify")], agent_replies=[text("QUESTION: Which flow do you mean?")])
    result = Assistant(llm=llm).ask("classify it")
    assert result["needs_follow_up"]
    assert result["answer"] == "Which flow do you mean?"


# ---------------------------------------------------------------- general / out of scope / router failures
def test_general_route_uses_no_agents():
    llm = FakeLLM(routes=[route("general")], final_replies=["A DDoS is ..."])
    result = Assistant(llm=llm).ask("what is a ddos?")
    assert result["agents_run"] == []
    assert result["answer"] == "A DDoS is ..."


def test_out_of_scope_gets_fixed_reply_without_llm():
    llm = FakeLLM(routes=[route("out_of_scope")])
    result = Assistant(llm=llm).ask("what's the weather?")
    assert result["answer"] == OUT_OF_SCOPE_REPLY
    assert llm.final_inputs == []


@pytest.mark.parametrize("router_output, expected", [
    ("refusal", "out_of_scope"),
    ("garbage", "general"),
    (RuntimeError("API down"), "general"),
])
def test_router_failures_never_crash(router_output, expected):
    llm = FakeLLM(routes=[router_output], final_replies=["ok"])
    result = Assistant(llm=llm).ask("something")
    assert result["route"] == expected


def test_router_falls_back_to_structured_output_when_plain_reply_is_invalid():
    llm = FakeLLM(routes=["garbage", route("search")], agent_replies=[text("nothing found")], final_replies=["ok"])
    result = Assistant(llm=llm).ask("has this happened before?")
    assert result["route"] == "search"
    assert len(llm.router_inputs) == 2  # plain attempt, then structured output


def test_router_plain_request_asks_for_json():
    llm = FakeLLM(routes=[route("general")], final_replies=["ok"])
    Assistant(llm=llm).ask("hi")
    assert '"route"' in llm.router_inputs[0][0].content


# ---------------------------------------------------------------- robustness
def test_unknown_tool_and_bad_args_become_error_evidence():
    llm = FakeLLM(
        routes=[route("classify")],
        agent_replies=[
            tool_call("hack_the_planet", {}, "c1"),
            tool_call("classify_attack_type", {"flow_duration": -5, "total_fwd_packets": 1}, "c2"),
            text("could not classify"),
        ],
    )
    result = Assistant(llm=llm).ask("classify")
    assert [e["result"]["status"] for e in result["evidence"]] == ["error", "error"]
    assert "Unknown tool" in result["evidence"][0]["result"]["error"]
    assert result["answer"] == "could not classify"


def test_tool_loop_is_bounded():
    endless = [tool_call("list_attack_types", {}, f"c{i}") for i in range(MAX_TOOL_ROUNDS)]
    llm = FakeLLM(
        routes=[route("classify")],
        agent_replies=[*endless, text("summary after budget")],
        final_replies=["done"],
    )
    result = Assistant(llm=llm).ask("loop forever")
    assert len(result["evidence"]) == MAX_TOOL_ROUNDS
    assert result["findings"]["classification"] == "summary after budget"


def test_empty_final_reply_falls_back_to_findings():
    llm = FakeLLM(routes=[route("classify")], agent_replies=[text("raw findings")], final_replies=[""])
    result = Assistant(llm=llm, direct_reply=False).ask("classify INC-000001")
    assert "raw findings" in result["answer"]


def test_empty_message_rejected():
    with pytest.raises(ValueError):
        Assistant(llm=FakeLLM()).ask("   ")


# ---------------------------------------------------------------- helpers
def test_next_step():
    assert next_step({"plan": ["classification", "response"], "completed": []}) == "agents"
    assert next_step({"plan": ["classification", "response"], "completed": ["classification"]}) == "agents"
    assert next_step({"plan": ["classification"], "completed": ["classification"]}) == "final_answer"
    assert next_step({"plan": ["classification"], "completed": [], "follow_up": "q?"}) == "ask_follow_up"


def test_plan_stages_and_next_stage():
    plan = ["classification", "risk_priority", "incident_search", "response"]
    assert plan_stages("investigate", plan) == [["classification", "risk_priority"], ["incident_search", "response"]]
    assert plan_stages("investigate", plan, parallel=False) == [[a] for a in plan]
    assert plan_stages("risk", ["risk_priority"]) == [["risk_priority"]]
    state = {"route": "investigate", "plan": plan, "completed": []}
    assert next_stage(state) == ["classification", "risk_priority"]
    assert next_stage({**state, "completed": ["classification", "risk_priority"]}) == ["incident_search", "response"]
    assert next_stage({**state, "completed": plan}) == []


def test_build_task_includes_history_and_findings():
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage("first"), AIMessage("answer"), HumanMessage("second")],
        "findings": {"classification": "PortScan"},
        "flow_args": {"incident_id": "INC-000005"},
    }
    task = build_task(state)
    assert "User: first" in task and "Assistant: answer" in task
    assert "Latest user message:\nsecond" in task
    assert "[Classification agent]\nPortScan" in task
    assert "INC-000005" in task


# ---------------------------------------------------------------- agent trace
def test_trace_records_route_agents_tools_and_final():
    llm = FakeLLM(
        routes=[route("classify")],
        agent_replies=[tool_call("classify_attack_type", {"incident_id": "INC-004393"}), text("findings")],
        final_replies=["answer"],
    )
    trace = Assistant(llm=llm).ask("What attack is INC-004393?")["trace"]

    assert [e["type"] for e in trace] == [
        "guard_input", "route", "agent_start", "tool_call", "tool_result", "agent_end", "final", "guard_output"
    ]
    assert trace[0]["action"] == "allowed"
    assert trace[1]["route"] == "classify" and trace[1]["plan"] == ["classification"]
    assert trace[4]["summary"] == "INC-004393: Infiltration (90%)"
    assert trace[5]["llm_calls"] == 2 and trace[5]["tool_calls"] == 1 and trace[5]["outcome"] == "findings"
    assert all(e["elapsed_ms"] >= 0 for e in trace if "elapsed_ms" in e)


def test_trace_resets_each_turn_and_records_follow_up():
    llm = FakeLLM(
        routes=[route("risk"), route("general")],
        agent_replies=[tool_call("assess_intrusion_risk", {"flow_duration": 5})],
        final_replies=["hi"],
    )
    assistant = Assistant(llm=llm)
    thread = assistant.new_thread_id()
    first = assistant.ask("suspicious? 5 us", thread)["trace"]
    assert first[-3]["outcome"] == "follow_up"
    assert first[-2]["type"] == "follow_up"
    assert first[-1]["type"] == "guard_output"
    assert first[4]["summary"].startswith("needs follow-up")

    second = assistant.ask("hello", thread)["trace"]
    assert [e["type"] for e in second] == ["guard_input", "route", "final", "guard_output"]


def test_stream_yields_live_events_then_result():
    llm = FakeLLM(
        routes=[route("classify")],
        agent_replies=[tool_call("list_attack_types", {}), text("answer")],
    )
    items = list(Assistant(llm=llm).stream("list attack types"))
    kinds = [kind for kind, _ in items]
    assert kinds[-1] == "result" and kinds.count("result") == 1
    events = [payload["type"] for kind, payload in items if kind == "event"]
    assert events == [
        "guard_input", "route", "agent_start", "tool_call", "tool_result", "agent_end", "final", "guard_output"
    ]
    assert items[-1][1]["answer"] == "answer"
    assert [e["type"] for e in items[-1][1]["trace"]] == events


@pytest.mark.parametrize("tool, args, expected", [
    ("classify_attack_type", {"incident_id": "INC-000035"}, "INC-000035: SSH-Patator (100%)"),
    ("assess_intrusion_risk", {"incident_id": "INC-000035"}, "risk 100% Critical, anomaly Low"),
    ("explain_flow", {"incident_id": "INC-002545"}, "Heartbleed (100%), risk Critical, anomaly High, P1"),
    ("prioritize_flow", {"incident_id": "INC-000005"}, "P3 Medium"),
    ("prioritize_flows", {"incident_ids": ["INC-000001", "INC-002545"]}, "most urgent: INC-002545"),
    ("get_attack_profile", {"attack_type": "ddos"}, "DDoS profile (severity High)"),
    ("list_attack_types", {}, "15 attack types"),
    ("lookup_incident", {"incident_id": "INC-000042"}, "historical label BENIGN, port 443"),
    ("classify_attack_type", {"flow_duration": -1}, "error:"),
    ("classify_attack_type", {"flow_duration": 5}, "needs follow-up: missing packet counts"),
])
def test_tool_result_summaries(tool, args, expected):
    from agents.tools import ALL_TOOLS
    from agents.trace import summarize_tool_result

    result = {t.name: t for t in ALL_TOOLS}[tool].invoke(args)
    assert expected in summarize_tool_result(tool, result)


def test_format_event_covers_all_event_types():
    from agents.trace import format_event

    events = [
        {"type": "route", "route": "investigate", "reason": "r", "plan": ["classification", "response"], "elapsed_ms": 900},
        {"type": "agent_start", "agent": "classification"},
        {"type": "tool_call", "agent": "classification", "tool": "x", "args": {}},
        {"type": "tool_result", "agent": "classification", "tool": "explain_flow", "args": {"incident_id": "INC-1"},
         "status": "ok", "summary": "s", "elapsed_ms": 10},
        {"type": "agent_end", "agent": "classification", "outcome": "findings", "llm_calls": 2, "tool_calls": 1, "elapsed_ms": 5},
        {"type": "follow_up", "question": "q"},
        {"type": "final", "findings": 2, "elapsed_ms": 3},
    ]
    lines = [format_event(e) for e in events]
    assert lines[2] is None  # tool_call is folded into tool_result
    assert "Classification agent -> Response agent" in lines[0]
    assert "explain_flow(incident_id=INC-1)" in lines[3]
    assert all(line for i, line in enumerate(lines) if i != 2)
