"""Deterministic synthetic trace generator for meeting_prep tool-calling."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tool_recovery_lora.data.schema import (
    TRACE_SCHEMA_VERSION,
    Message,
    ToolCall,
    TraceExample,
)

TOOL_NAME = "meeting_prep"
WRONG_TOOLS = ("research_analysis", "summariser", "executive_brief", "support_triage")
ErrorType = Literal["missing_args", "wrong_tool", "invalid_json"]

SYSTEM_PROMPT = (
    "You are a sales meeting-prep agent for langgraph-agent-stack. "
    "When the user asks for a meeting brief, call the meeting_prep tool with "
    "valid JSON arguments: company (required), person, meeting_goal, context. "
    "If a tool call fails, repair it and try again."
)

COMPANIES = [
    "Acme Corp",
    "Globex",
    "Initech",
    "Umbrella Health",
    "Stark Industries",
    "Wayne Enterprises",
    "Hooli",
    "Pied Piper",
    "Massive Dynamic",
    "Soylent Systems",
    "BlueSun",
    "Cyberdyne",
    "Tyrell Analytics",
    "Oscorp",
    "LexCorp",
    "Aperture Labs",
    "Black Mesa",
    "Rekall",
    "Wonka Foods",
    "Duff Energy",
    "Vandelay Industries",
    "Prestige Worldwide",
    "Nakatomi Trading",
    "Gekko Capital",
    "Buy n Large",
]

PEOPLE = [
    "Jane Doe",
    "Bob Martinez",
    "Alice Nguyen",
    "Sam Okonkwo",
    "Priya Shah",
    "Chris Alvarez",
    "Morgan Lee",
    "Taylor Brooks",
    "Jordan Kim",
    "Riley Chen",
    "Alex Rivera",
    "Casey Patel",
    "Jamie Torres",
    "Avery Quinn",
    "Drew Hassan",
    "Cameron Blake",
    "Reese Okada",
    "Skyler Berg",
    "Parker Singh",
    "Quinn Morales",
]

GOALS = [
    "discovery",
    "discovery call",
    "partnership",
    "renewal",
    "upsell",
    "technical deep-dive",
    "pricing negotiation",
    "executive alignment",
    "pilot kickoff",
    "QBR prep",
    "security review",
    "onboarding checkpoint",
    "expansion discussion",
    "competitive displacement",
    "budget planning",
]

CONTEXTS = [
    "",
    "They use Salesforce today.",
    "Warm intro from an existing customer.",
    "Renewal in 60 days.",
    "Competing with a legacy vendor.",
    "First meeting after a conference booth visit.",
    "Stakeholder is skeptical about AI features.",
    "Need to address a recent support incident.",
    "They asked for SOC2 evidence.",
    "Multi-region rollout under discussion.",
]

USER_TEMPLATES = [
    "Prepare a briefing for {company} with {person} about {goal}.",
    "I need meeting prep: company={company}, contact={person}, goal={goal}.",
    "Can you run meeting_prep for {person} at {company}? Goal: {goal}.",
    "Brief me before my call with {person} ({company}). Focus on {goal}.",
    "Meeting tomorrow with {person} from {company} — goal is {goal}. Prep me.",
    "Generate a sales brief for {company} / {person}, meeting goal: {goal}.",
    "Help me prepare: {company}, talking to {person}, objective={goal}.",
    "Please call the meeting prep tool for {company} and {person} ({goal}).",
]


@dataclass(frozen=True)
class Scenario:
    """One ground-truth meeting_prep invocation."""

    company: str
    person: str
    meeting_goal: str
    context: str

    def as_arguments(self) -> dict[str, Any]:
        """Return tool arguments matching MeetingPrepInput fields."""
        args: dict[str, Any] = {
            "company": self.company,
            "person": self.person,
            "meeting_goal": self.meeting_goal,
        }
        if self.context:
            args["context"] = self.context
        return args


def _user_text(rng: random.Random, scenario: Scenario) -> str:
    template = rng.choice(USER_TEMPLATES)
    text = template.format(
        company=scenario.company,
        person=scenario.person,
        goal=scenario.meeting_goal,
    )
    # Always surface non-empty context in the user turn so labels are learnable.
    if scenario.context:
        text = f"{text} Context: {scenario.context}"
    return text


def _scenario(rng: random.Random) -> Scenario:
    return Scenario(
        company=rng.choice(COMPANIES),
        person=rng.choice(PEOPLE),
        meeting_goal=rng.choice(GOALS),
        context=rng.choice(CONTEXTS),
    )


def _correct_example(example_id: str, rng: random.Random) -> TraceExample:
    scenario = _scenario(rng)
    expected = ToolCall(name=TOOL_NAME, arguments=scenario.as_arguments())
    return TraceExample(
        id=example_id,
        schema_version=TRACE_SCHEMA_VERSION,
        kind="correct",
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=_user_text(rng, scenario)),
            Message(role="assistant", content=None, tool_calls=[expected]),
        ],
        meta={
            "pack_id": TOOL_NAME,
            "expected_tool_call": expected.model_dump(),
            "source": "synthetic_v1",
        },
    )


def _broken_missing_args(scenario: Scenario, rng: random.Random) -> ToolCall:
    args = scenario.as_arguments()
    mode = rng.choice(["drop_company", "drop_goal", "empty_company"])
    if mode == "drop_company":
        args.pop("company", None)
    elif mode == "drop_goal":
        args.pop("meeting_goal", None)
    else:
        args["company"] = ""
    return ToolCall(name=TOOL_NAME, arguments=args)


def _broken_wrong_tool(scenario: Scenario, rng: random.Random) -> ToolCall:
    wrong = rng.choice(WRONG_TOOLS)
    # Mimic a confused router: stuff a query-like payload.
    return ToolCall(
        name=wrong,
        arguments={
            "query": (
                f"{scenario.company} {scenario.person} {scenario.meeting_goal}"
            )
        },
    )


def _recovery_example(
    example_id: str,
    rng: random.Random,
    error_type: ErrorType,
) -> TraceExample:
    scenario = _scenario(rng)
    expected = ToolCall(name=TOOL_NAME, arguments=scenario.as_arguments())
    user = Message(role="user", content=_user_text(rng, scenario))
    system = Message(role="system", content=SYSTEM_PROMPT)

    if error_type == "invalid_json":
        bad_content = (
            f'{{"name": "{TOOL_NAME}", "arguments": '
            f'{{"company": "{scenario.company}", "person": '
            f'"{scenario.person}"'  # deliberately truncated JSON
        )
        messages = [
            system,
            user,
            Message(role="assistant", content=bad_content),
            Message(
                role="tool",
                tool_call_id="call_invalid_json",
                name=TOOL_NAME,
                content=(
                    "ERROR: invalid_json — expected a structured tool call, "
                    "got malformed JSON text"
                ),
            ),
            Message(role="assistant", content=None, tool_calls=[expected]),
        ]
    elif error_type == "wrong_tool":
        broken = _broken_wrong_tool(scenario, rng)
        messages = [
            system,
            user,
            Message(role="assistant", content=None, tool_calls=[broken]),
            Message(
                role="tool",
                tool_call_id="call_wrong_tool",
                name=broken.name,
                content=(
                    f"ERROR: wrong_tool — unknown tool for this task; "
                    f"use {TOOL_NAME}"
                ),
            ),
            Message(role="assistant", content=None, tool_calls=[expected]),
        ]
    else:  # missing_args
        broken = _broken_missing_args(scenario, rng)
        messages = [
            system,
            user,
            Message(role="assistant", content=None, tool_calls=[broken]),
            Message(
                role="tool",
                tool_call_id="call_missing_args",
                name=TOOL_NAME,
                content=(
                    "ERROR: missing_args — company is required and "
                    "meeting_goal should be set when provided by the user"
                ),
            ),
            Message(role="assistant", content=None, tool_calls=[expected]),
        ]

    return TraceExample(
        id=example_id,
        schema_version=TRACE_SCHEMA_VERSION,
        kind="recovery",
        messages=messages,
        meta={
            "pack_id": TOOL_NAME,
            "error_type": error_type,
            "expected_tool_call": expected.model_dump(),
            "source": "synthetic_v1",
        },
    )


def generate_examples(
    n_total: int,
    *,
    seed: int = 42,
    correct_ratio: float = 0.6,
    id_prefix: str = "syn",
) -> list[TraceExample]:
    """Generate a mixed curriculum of correct and recovery traces.

    Args:
        n_total: Number of examples to generate.
        seed: RNG seed for reproducibility.
        correct_ratio: Fraction of ``correct`` examples (rest are recovery).
        id_prefix: Prefix for example ids.

    Returns:
        List of validated ``TraceExample`` instances.
    """
    if n_total < 1:
        raise ValueError("n_total must be >= 1")
    if not 0.0 <= correct_ratio <= 1.0:
        raise ValueError("correct_ratio must be in [0, 1]")

    rng = random.Random(seed)
    n_correct = int(round(n_total * correct_ratio))
    n_recovery = n_total - n_correct
    error_types: list[ErrorType] = ["missing_args", "wrong_tool", "invalid_json"]

    examples: list[TraceExample] = []
    for index in range(n_correct):
        examples.append(_correct_example(f"{id_prefix}-correct-{index:04d}", rng))
    for index in range(n_recovery):
        error_type = error_types[index % len(error_types)]
        examples.append(
            _recovery_example(
                f"{id_prefix}-recovery-{index:04d}",
                rng,
                error_type,
            )
        )
    rng.shuffle(examples)
    return examples


def split_train_val_eval(
    examples: list[TraceExample],
    *,
    n_eval: int,
    n_val: int,
    seed: int = 42,
) -> tuple[list[TraceExample], list[TraceExample], list[TraceExample]]:
    """Split examples into train / val / eval with disjoint ids.

    Args:
        examples: Full generated pool.
        n_eval: Smoke/eval set size.
        n_val: Validation set size.
        seed: Shuffle seed.

    Returns:
        ``(train, val, eval_set)``.

    Raises:
        ValueError: If the pool is too small.
    """
    if n_eval + n_val >= len(examples):
        raise ValueError("n_eval + n_val must be < len(examples)")
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    eval_set = shuffled[:n_eval]
    val = shuffled[n_eval : n_eval + n_val]
    train = shuffled[n_eval + n_val :]
    return train, val, eval_set


def write_jsonl(path: Path, examples: list[TraceExample]) -> None:
    """Write examples as JSONL (one object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.model_dump(), ensure_ascii=False))
            handle.write("\n")
