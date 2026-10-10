"""
Rule-based guardrails around the assistant graph (no LLM calls, deterministic).

Input guard (before the supervisor):
    - trims over-long messages
    - blocks prompt-injection attempts and requests to build malware / attack systems
    - redacts secrets and personal data (API keys, passwords, private keys, card numbers, emails)

Output guard (after the final answer or follow-up question):
    - marks incident IDs that appear in no tool result or message as unverified
    - flags percentages and priorities that do not match the tool results
    - removes dangerous commands (rm -rf /, disabling firewalls / antivirus, ...)
    - adds a disclaimer when the analysis rests on partial metrics or low confidence
    - adds a note when the model prediction differs from an incident's historical label
    - redacts secrets and blocks leaks of the system prompts

The router's out_of_scope route (model refusals) stays as a second, LLM-based layer.
"""
import functools
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

MAX_INPUT_CHARS = 4000

BLOCKED_REPLIES = {
    "prompt_injection": (
        "I can't follow instructions that try to change my rules or reveal my internal instructions. "
        "I can classify network flows, assess their risk and priority, find similar historical incidents, "
        "and recommend investigation and response steps."
    ),
    "harmful_request": (
        "I can't help create malware, attack systems or evade security controls. I can help you detect, "
        "investigate and respond to such attacks: classify flows, assess risk and priority, find similar "
        "historical incidents, and recommend containment and remediation steps."
    ),
}
BLOCKED_MESSAGE_PLACEHOLDER = "[message blocked by input guardrail: {category}]"
PROMPT_LEAK_REPLY = "I can't share my internal instructions. Ask me about a network flow or incident instead."


# ============================================================================
# Patterns
# ============================================================================
_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")


def normalize_text(text: str) -> str:
    """Lowercase, NFKC-normalize, drop zero-width characters and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()


INJECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("ignore instructions", re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,30}\b(previous|prior|above|earlier|all|your|the|system)\b"
        r".{0,20}\b(instructions?|rules?|prompts?|guidelines?|directives?|guardrails?|restrictions?)\b")),
    ("reveal system prompt", re.compile(
        r"\b(reveal|show|print|repeat|output|display|tell me|give me|leak|what (is|are))\b.{0,30}"
        r"\b(your|the)\b.{0,15}\b(system prompt|system message|hidden prompt|initial prompt|instructions you were given|"
        r"internal instructions|developer (message|prompt))\b")),
    ("role override", re.compile(
        r"\b(you are now|from now on,? you are|pretend (to be|you are)|act as)\b.{0,40}"
        r"\b(dan|unrestricted|unfiltered|jailbroken|evil|no (rules|restrictions|limits))\b")),
    ("jailbreak keyword", re.compile(r"\b(jailbreak|developer mode|dan mode|do anything now)\b")),
    ("fake role tag", re.compile(r"<\|?\s*(system|im_start|im_end)\s*\|?>|\[\s*(system|inst)\s*\]")),
    ("new instructions", re.compile(r"\b(new|updated|real) (instructions|rules|system prompt)\s*:")),
]
# Checked on the raw text: a line that pretends to be a system / assistant turn.
FAKE_TURN = re.compile(r"(?im)^\s*(system|assistant|developer)\s*:")

# Words that make a request about defending, not attacking ("how do I detect ransomware?").
_DEFENSIVE = re.compile(
    r"\b(detect\w*|defen[cs]\w*|defend\w*|prevent\w*|protect\w*|mitigat\w*|respond\w*|response|identify\w*|"
    r"block\w*|investigat\w*|analy[sz]\w*|recogni[sz]\w*|remov\w*|contain\w*|remediat\w*|stop\w*|signs? of|"
    r"indicators?|ioc|recover\w*|hunt\w*|monitor\w*|alert\w*|rules?|signatures?|report|summar\w*|explain\w*|"
    r"overview|about|history|describe|description|awareness|training|policy|playbook)\b"
)
_MALWARE = (
    r"(malware|ransomware|keylogger|key logger|trojan|rootkit|botnet|virus|worm|spyware|backdoor|reverse shell|"
    r"rat\b|remote access trojan|exploit( code)?|shellcode|cryptominer|stealer|dropper)"
)
# "How do I ..." style: the user wants to do it themselves (not "how do attackers ...").
_HOW_I = r"\b(how (do|can|would|should) (i|we)|how to|help me|i want to|i need to|teach me|show me how to|let'?s)\b"
HARMFUL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("build malware", re.compile(
        r"\b(write|create|build|code|develop|generate|program|craft|send me|make (me )?(a|an|some|my own)|"
        r"give me (a|an|some|the code|code|working))\b.{0,40}\b" + _MALWARE)),
    ("attack a target", re.compile(
        _HOW_I + r".{0,20}\b(launch|perform|run|execute|carry out|start|conduct|do)\b.{0,25}\b(ddos|dos attack|"
        r"denial of service|brute[- ]?force|sql injection|xss attack|port ?scan|attack)\b.{0,25}\b(against|on|at)\b")),
    ("hack into", re.compile(_HOW_I + r".{0,20}\b(hack|break) into\b")),
    ("evade detection", re.compile(
        _HOW_I + r".{0,20}\b(bypass|evade|avoid|disable|get past|hide from|fool|turn off)\b.{0,30}"
        r"\b(antivirus|anti-virus|edr|ids|ips|firewall|detection|defender|siem|av|security controls?)\b")),
    ("steal credentials", re.compile(
        _HOW_I + r".{0,20}\b(steal|harvest|dump|exfiltrate|phish|crack)\b.{0,30}\b(credentials?|passwords?|"
        r"hashes|tokens?|cookies|sessions?|someone'?s|their)\b")),
]

# Secrets and personal data: (kind, pattern, replacement group or None for the whole match).
SECRET_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|$)", re.S)),
    ("api_key", re.compile(r"\bsk-(ant-)?[A-Za-z0-9_\-]{16,}")),
    ("api_key", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("api_key", re.compile(r"\b(ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("api_key", re.compile(r"\b(xox[abpr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_\-]{35})\b")),
    ("bearer_token", re.compile(r"\bbearer\s+[A-Za-z0-9._\-]{20,}", re.I)),
    ("password", re.compile(r"\b(password|passwd|pwd|passphrase)\b\s*(is|:|=)\s*(?P<secret>\S+)", re.I)),
    ("api_key", re.compile(r"\b(api[_ -]?key|secret|token|access[_ -]?key)\b\s*(is|:|=)\s*(?P<secret>\S{8,})", re.I)),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]
_CARD_CANDIDATE = re.compile(r"\b(?:\d{4}[ -]?){3}\d{1,7}\b")


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_sensitive(text: str) -> Tuple[str, List[str]]:
    """Replace secrets and personal data with [REDACTED_<KIND>]; return the text and the kinds found."""
    found: List[str] = []

    for kind, pattern in SECRET_PATTERNS:
        def replace(match: re.Match, kind=kind) -> str:
            found.append(kind)
            tag = f"[REDACTED_{kind.upper()}]"
            if "secret" in match.re.groupindex:
                start, end = match.span("secret")
                offset = match.start()
                whole = match.group(0)
                return whole[: start - offset] + tag + whole[end - offset:]
            return tag

        text = pattern.sub(replace, text)

    def replace_card(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("card_number")
            return "[REDACTED_CARD_NUMBER]"
        return match.group(0)

    text = _CARD_CANDIDATE.sub(replace_card, text)
    return text, sorted(set(found))


# ============================================================================
# Input guard
# ============================================================================
@dataclass
class InputCheck:
    action: str                       # allowed | redacted | blocked
    text: str                         # message to pass on (trimmed / redacted)
    category: Optional[str] = None    # prompt_injection | harmful_request when blocked
    matched: Optional[str] = None     # name of the rule that matched
    redactions: List[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def blocked(self) -> bool:
        return self.action == "blocked"

    @property
    def reply(self) -> Optional[str]:
        return BLOCKED_REPLIES.get(self.category) if self.blocked else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "category": self.category,
            "matched": self.matched,
            "redactions": self.redactions,
            "truncated": self.truncated,
        }


def _first_match(text: str, patterns: Iterable[Tuple[str, re.Pattern]]) -> Optional[str]:
    for name, pattern in patterns:
        if pattern.search(text):
            return name
    return None


def check_input(message: str) -> InputCheck:
    """Classify a user message as allowed, redacted or blocked."""
    text = (message or "").strip()
    truncated = len(text) > MAX_INPUT_CHARS
    if truncated:
        text = text[:MAX_INPUT_CHARS]

    normalized = normalize_text(text)
    rule = _first_match(normalized, INJECTION_PATTERNS) or ("fake role tag" if FAKE_TURN.search(text) else None)
    if rule:
        return InputCheck("blocked", text, "prompt_injection", rule, truncated=truncated)

    rule = _first_match(normalized, HARMFUL_PATTERNS)
    if rule and not _DEFENSIVE.search(normalized):
        return InputCheck("blocked", text, "harmful_request", rule, truncated=truncated)

    text, redactions = redact_sensitive(text)
    action = "redacted" if redactions else "allowed"
    return InputCheck(action, text, redactions=redactions, truncated=truncated)


# ============================================================================
# Output guard
# ============================================================================
INCIDENT_ID = re.compile(r"\bINC-\d{6}\b", re.I)
PERCENT = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s?%")
PRIORITY = re.compile(r"\bP([1-4])\b")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
PERCENT_TOLERANCE = 1.0  # percentage points: allows rounding (0.9996 -> "100%")

DANGEROUS_COMMANDS: List[Tuple[str, re.Pattern]] = [
    ("deletes the whole filesystem", re.compile(r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r|-r\s+-f|-f\s+-r)[a-z]*\s+(/|~|\*|/\*)(\s|$)", re.I)),
    ("formats a disk", re.compile(r"\b(mkfs(\.\w+)?\s+/dev/|format\s+[a-z]:|diskpart\b.*\bclean\b)", re.I)),
    ("overwrites a disk", re.compile(r"\bdd\s+if=\S+\s+of=/dev/(sd|nvme|hd|disk)", re.I)),
    ("fork bomb", re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")),
    ("opens all file permissions", re.compile(r"\bchmod\s+(-R\s+)?777\s+/(\s|$)", re.I)),
    ("deletes Windows system files", re.compile(r"\b(del|rd|rmdir)\s+(/[sqf]\s+)+[a-z]:\\(windows|\s|$)", re.I)),
    ("flushes all firewall rules", re.compile(r"\b(iptables\s+(-F|--flush)(\s*$|\s+[^\w-])|nft\s+flush\s+ruleset)", re.I)),
    ("disables the firewall", re.compile(
        r"\b(ufw\s+disable|systemctl\s+(stop|disable)\s+(firewalld|ufw|iptables)|"
        r"netsh\s+advfirewall\s+set\s+\w+\s+state\s+off|Set-NetFirewallProfile\b.*-Enabled\s+(False|0))", re.I)),
    ("disables antivirus", re.compile(
        r"\b(Set-MpPreference\b.*-Disable\w+\s+(\$true|1)|sc\s+(stop|config)\s+windefend|"
        r"systemctl\s+(stop|disable)\s+(clamav|falcon|crowdstrike|auditd))", re.I)),
    ("disables SELinux", re.compile(r"\bsetenforce\s+0\b", re.I)),
    ("pipes a download into a shell", re.compile(r"\b(curl|wget)\b[^|\n]*\|\s*(sudo\s+)?(ba|z)?sh\b", re.I)),
]

DISCLAIMERS = {
    "low": (
        "Confidence note: this analysis is based on partial flow metrics or a low-confidence prediction. "
        "Treat it as indicative only and confirm with the full flow record."
    ),
    "reduced": (
        "Confidence note: some flow metrics were missing and filled with normal-traffic values, so "
        "confidence is reduced."
    ),
}
# The answer already states the caveat in its own words (the final answer prompt asks it to).
_DISCLAIMER_ALREADY = re.compile(
    r"\b(indicative|partial (flow )?(metrics|input|data)|reduced(-| )confidence|less reliable|"
    r"confidence is (reduced|low|medium)|input quality (is |was |: ?)?(reduced|low)|(reduced|low) input quality)\b",
    re.I,
)

UNVERIFIED_MARK = "(unverified ID)"


@dataclass
class OutputCheck:
    text: str
    issues: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {"issues": self.issues, "changed": self.changed}


def _walk(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v)
    else:
        yield value


def _evidence_results(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e.get("result") or {} for e in evidence]


def _known_ids(evidence: List[Dict[str, Any]], context_texts: Iterable[str]) -> Set[str]:
    blob = json.dumps(evidence, default=str) + "\n" + "\n".join(context_texts)
    return {m.upper() for m in INCIDENT_ID.findall(blob)}


def _known_numbers(evidence: List[Dict[str, Any]], context_texts: Iterable[str]) -> List[float]:
    numbers: List[float] = []
    for value in _walk(_evidence_results(evidence)):
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numbers.append(float(value))
            if 0 <= value <= 1:
                numbers.append(float(value) * 100)
        elif isinstance(value, str):
            numbers.extend(float(n) for n in NUMBER.findall(value))
    for text in context_texts:
        numbers.extend(float(n) for n in NUMBER.findall(text))
    return numbers


def _evidence_priorities(evidence: List[Dict[str, Any]]) -> Set[str]:
    found: Set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "priority" and isinstance(item, str) and re.fullmatch(r"P[1-4]", item):
                    found.add(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(_evidence_results(evidence))
    return found


def _has_model_results(evidence: List[Dict[str, Any]]) -> bool:
    return any(r.get("status") == "ok" and ("attack_type" in r or "risk" in r) for r in _evidence_results(evidence))


def _check_ids(text: str, known: Set[str], issues: List[Dict[str, Any]]) -> str:
    unverified = sorted({m.upper() for m in INCIDENT_ID.findall(text)} - known)
    if not unverified:
        return text
    issues.append({"check": "unverified_incident_ids", "ids": unverified, "action": "marked as unverified"})
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, unverified)) + r")\b(?!\s*\(unverified)", re.I)
    return pattern.sub(lambda m: f"{m.group(1)} {UNVERIFIED_MARK}", text)


def _check_numbers(text: str, evidence: List[Dict[str, Any]], context: List[str], issues: List[Dict[str, Any]]) -> List[str]:
    if not _has_model_results(evidence):
        return []
    known = _known_numbers(evidence, context)
    stated = {float(v) for v in PERCENT.findall(text)}
    unmatched = sorted(v for v in stated if not any(abs(v - k) <= PERCENT_TOLERANCE for k in known))
    notes = []
    if unmatched:
        shown = ", ".join(f"{v:g}%" for v in unmatched)
        issues.append({"check": "unmatched_percentages", "values": unmatched, "action": "flagged"})
        notes.append(f"Check: {shown} could not be matched to the model outputs; verify before acting on it.")

    priorities = _evidence_priorities(evidence)
    mentioned = {f"P{p}" for p in PRIORITY.findall(text)}
    if priorities and mentioned and not (mentioned & priorities):
        issues.append({"check": "priority_mismatch", "stated": sorted(mentioned), "tools": sorted(priorities),
                       "action": "flagged"})
        notes.append(f"Check: the prioritization tool returned {', '.join(sorted(priorities))}, "
                     f"not {', '.join(sorted(mentioned))}.")
    return notes


def _check_commands(text: str, issues: List[Dict[str, Any]]) -> str:
    lines = text.split("\n")
    removed = []
    for i, line in enumerate(lines):
        for reason, pattern in DANGEROUS_COMMANDS:
            if pattern.search(line):
                removed.append(reason)
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = f"{indent}[command removed by safety guardrail: {reason}]"
                break
    if removed:
        issues.append({"check": "dangerous_commands", "reasons": removed, "action": "removed"})
    return "\n".join(lines)


def _confidence_note(text: str, evidence: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> Optional[str]:
    level = None
    for result in _evidence_results(evidence):
        if result.get("status") != "ok":
            continue
        quality = (result.get("input") or {}).get("input_quality")
        if quality == "low" or result.get("overall_confidence") == "low":
            level = "low"
            break
        if quality == "reduced":
            level = "reduced"
    if level is None or _DISCLAIMER_ALREADY.search(text):
        return None
    issues.append({"check": "confidence_disclaimer", "level": level, "action": "added"})
    return DISCLAIMERS[level]


def _label_notes(text: str, evidence: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> List[str]:
    notes, seen = [], set()
    lowered = text.lower()
    for result in _evidence_results(evidence):
        incident, attack = result.get("incident"), result.get("attack_type")
        if result.get("status") != "ok" or not incident or not isinstance(attack, dict):
            continue
        label, predicted = incident.get("historical_label"), attack.get("prediction")
        key = incident.get("incident_id")
        if not label or not predicted or label == predicted or key in seen:
            continue
        seen.add(key)
        if label.lower() in lowered:
            continue
        issues.append({"check": "historical_label_mismatch", "incident_id": key, "historical_label": label,
                       "prediction": predicted, "action": "note added"})
        notes.append(f"Note: {key} is historically labelled {label}, but the model predicts {predicted}.")
    return notes


@functools.lru_cache(maxsize=1)
def _prompt_fragments() -> Tuple[str, ...]:
    from agents import prompts

    texts = [prompts.ROUTER_PROMPT, prompts.SHARED_AGENT_RULES, prompts.FINAL_ANSWER_PROMPT,
             *prompts.AGENT_PROMPTS.values()]
    fragments = set()
    for text in texts:
        for piece in re.split(r"\n|(?<=[.:;])\s", text):
            piece = normalize_text(piece.lstrip("- "))
            if len(piece) >= 60:
                fragments.add(piece)
    return tuple(sorted(fragments))


def check_output(
    answer: str,
    evidence: Optional[List[Dict[str, Any]]] = None,
    context_texts: Iterable[str] = (),
) -> OutputCheck:
    """Verify and, where needed, fix the final reply.

    Args:
        answer: reply produced by the final answer node (or a follow-up question).
        evidence: this turn's tool calls ({agent, tool, args, result}).
        context_texts: conversation messages; IDs and numbers the user gave count as known.
    """
    evidence = evidence or []
    context = list(context_texts)
    issues: List[Dict[str, Any]] = []
    text = answer or ""

    normalized = normalize_text(text)
    if any(fragment in normalized for fragment in _prompt_fragments()):
        issues.append({"check": "system_prompt_leak", "action": "reply replaced"})
        return OutputCheck(PROMPT_LEAK_REPLY, issues)

    text, redactions = redact_sensitive(text)
    if redactions:
        issues.append({"check": "sensitive_data", "kinds": redactions, "action": "redacted"})

    text = _check_commands(text, issues)
    text = _check_ids(text, _known_ids(evidence, context), issues)

    notes = _check_numbers(text, evidence, context, issues)
    notes += _label_notes(text, evidence, issues)
    disclaimer = _confidence_note(text, evidence, issues)
    if disclaimer:
        notes.append(disclaimer)
    if notes:
        text = text.rstrip() + "\n\n" + "\n".join(f"> {note}" for note in notes)
    return OutputCheck(text, issues)
