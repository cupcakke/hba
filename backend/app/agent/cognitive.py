from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Awaitable, Callable

from app import database as database_module
from app.errors import AgentFault

VECTOR_DIMS = (
    "confidence",
    "progress",
    "risk",
    "uncertainty",
    "complexity",
    "novelty",
    "urgency",
    "tool_need",
    "info_need",
    "verification_need",
    "reversibility",
    "resource_cost",
    "alignment",
    "coherence",
    "momentum",
    "blocker_pressure",
)

GATE_WEIGHTS: dict[str, float] = {
    "confidence": 1.6,
    "progress": 0.8,
    "risk": -1.4,
    "uncertainty": -1.2,
    "complexity": -0.4,
    "novelty": -0.3,
    "urgency": 0.3,
    "tool_need": 0.2,
    "info_need": -0.9,
    "verification_need": -0.3,
    "reversibility": 0.7,
    "resource_cost": -0.4,
    "alignment": 1.0,
    "coherence": 0.9,
    "momentum": 0.5,
    "blocker_pressure": -1.3,
}

GATE_BIAS = 0.1

PHASES = (
    "osPerceive",
    "osDeliberate",
    "osAct",
    "osValidate",
    "osReflect",
    "osConsolidate",
)

TRANSITIONS: dict[str, tuple[str, ...]] = {
    "osPerceive": ("osDeliberate", "osAct", "osConsolidate"),
    "osDeliberate": ("osAct", "osConsolidate"),
    "osAct": ("osAct", "osValidate", "osReflect", "osPerceive", "osDeliberate", "osConsolidate"),
    "osValidate": ("osReflect", "osConsolidate"),
    "osReflect": ("osDeliberate", "osConsolidate"),
    "osConsolidate": (),
}

ALLOWED_COMPONENTS = (
    "planner",
    "executor",
    "tool",
    "environment",
    "verifier",
    "model",
)

DSL_KEYWORDS = ("SKILL", "WHEN", "REQUIRE", "STEP", "VERIFY", "RECOVER")

REFLECTION_FIELDS = (
    "failure_point",
    "component",
    "root_cause",
    "pivot_action",
    "patch",
    "failed_action",
    "corrective_action",
)

EMBED_DIM = 256
REASON_MAX_DEPTH = 4
REASON_MAX_BRANCHES = 6
REASON_MAX_CALLS = 300

PROBE_TIMEOUT_SECONDS = 60.0
THINK_ATTEMPTS = 3
THINK_BASE_DELAY = 0.5
POLICY_WEIGHT_BOUND = 5.0
POLICY_DECAY = 0.99
SKILL_RETIRE_MIN_TRIALS = 8
SKILL_RETIRE_UPPER_BOUND = 0.3
CLIP_MARKER = "\n[...]\n"

ATOMIC_REASONER_SYSTEM = (
    "Izolált következtető vagy. Tömören, tényszerűen válaszolj a kérdésre; "
    "ne javasolj állapotmódosítást."
)

Think = Callable[[str, str], Awaitable[str]]
Probe = Callable[[str], Awaitable[bool]]

_db_write_lock = asyncio.Lock()
_counter_lock = asyncio.Lock()
_subgoal_counter = 0
_JSON_DECODER = json.JSONDecoder()
_NOT_FOUND = object()


@dataclass
class Subgoal:
    id: str
    text: str
    done: bool = False


@dataclass
class GoalSpec:
    state_pointers: list[dict[str, Any]] = field(default_factory=list)
    required_files: list[str] = field(default_factory=list)
    file_contents: list[dict[str, str]] = field(default_factory=list)

    def copy(self) -> "GoalSpec":
        return GoalSpec(
            state_pointers=[dict(item) for item in self.state_pointers],
            required_files=list(self.required_files),
            file_contents=[dict(item) for item in self.file_contents],
        )


def to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    number = to_float(value, math.nan)
    if math.isnan(number):
        return default
    return int(number)


def to_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def to_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def to_file_contents(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [
        {"path": item["path"], "contains": item["contains"]}
        for item in value
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("contains"), str)
    ]


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


@dataclass
class Sigma:
    objective: str = ""
    strategy: str = ""
    subgoals: list[Subgoal] = field(default_factory=list)
    closed_subgoals: int = 0
    facts: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    vector: list[float] = field(default_factory=list)
    gate: float = 0.0
    decision: str = "act"
    goal: GoalSpec = field(default_factory=GoalSpec)

    def copy(self) -> "Sigma":
        return Sigma(
            objective=self.objective,
            strategy=self.strategy,
            subgoals=[Subgoal(item.id, item.text, item.done) for item in self.subgoals],
            closed_subgoals=self.closed_subgoals,
            facts=list(self.facts),
            blockers=list(self.blockers),
            next_steps=list(self.next_steps),
            vector=list(self.vector),
            gate=self.gate,
            decision=self.decision,
            goal=self.goal.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "strategy": self.strategy,
            "subgoals": [
                {"id": item.id, "text": item.text, "done": item.done} for item in self.subgoals
            ],
            "closed_subgoals": self.closed_subgoals,
            "facts": list(self.facts),
            "blockers": list(self.blockers),
            "next_steps": list(self.next_steps),
            "vector": list(self.vector),
            "gate": self.gate,
            "decision": self.decision,
            "goal": {
                "state_pointers": [dict(item) for item in self.goal.state_pointers],
                "required_files": list(self.goal.required_files),
                "file_contents": [dict(item) for item in self.goal.file_contents],
            },
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "Sigma":
        goal_raw = payload.get("goal")
        goal = goal_raw if isinstance(goal_raw, dict) else {}
        subgoals_raw = payload.get("subgoals")
        vector_raw = payload.get("vector")
        return Sigma(
            objective=str(payload.get("objective") or ""),
            strategy=str(payload.get("strategy") or ""),
            subgoals=[
                Subgoal(
                    str(item.get("id") or ""),
                    str(item.get("text") or ""),
                    bool(item.get("done")),
                )
                for item in (subgoals_raw if isinstance(subgoals_raw, list) else [])
                if isinstance(item, dict)
            ],
            closed_subgoals=max(0, to_int(payload.get("closed_subgoals"))),
            facts=to_str_list(payload.get("facts")),
            blockers=to_str_list(payload.get("blockers")),
            next_steps=to_str_list(payload.get("next_steps")),
            vector=[to_float(item, 0.5) for item in vector_raw] if isinstance(vector_raw, list) else [],
            gate=to_float(payload.get("gate")),
            decision=str(payload.get("decision") or "act"),
            goal=GoalSpec(
                state_pointers=to_dict_list(goal.get("state_pointers")),
                required_files=to_str_list(goal.get("required_files")),
                file_contents=to_file_contents(goal.get("file_contents")),
            ),
        )


@dataclass
class Reflection:
    failure_point: str
    component: str
    root_cause: str
    pivot_action: str
    patch: str
    failed_action: str
    corrective_action: str


@dataclass
class ReasonNode:
    question: str
    answer: str
    children: list["ReasonNode"]


@dataclass
class Skill:
    id: str
    name: str
    dsl: str
    terms: str
    vector: list[float]
    successes: int
    failures: int


def clip(value: str, max_length: int) -> str:
    if max_length <= 0:
        return ""
    if len(value) <= max_length:
        return value
    return value[:max_length]


def clip_middle(value: str, max_length: int, head_ratio: float = 0.25) -> str:
    if max_length <= 0:
        return ""
    if len(value) <= max_length:
        return value
    if max_length <= len(CLIP_MARKER) + 2:
        return value[-max_length:]
    budget = max_length - len(CLIP_MARKER)
    head = int(budget * clamp(head_ratio, 0.0, 1.0))
    tail = budget - head
    return value[:head] + CLIP_MARKER + value[len(value) - tail :]


def is_record(value: Any) -> bool:
    return isinstance(value, dict)


def strip_sensitive(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_sensitive(item) for item in value]
    if not is_record(value):
        return value
    out: dict[str, Any] = {}
    for key, inner in value.items():
        if key not in ("messages", "chain_of_thought"):
            out[key] = strip_sensitive(inner)
    return out


def normalize_vector(vector: list[float]) -> list[float]:
    result: list[float] = []
    for index in range(len(VECTOR_DIMS)):
        raw = vector[index] if index < len(vector) else 0.5
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raw = 0.5
        result.append(clamp(float(raw), 0.0, 1.0))
    return result


def gate(vector: list[float]) -> tuple[float, str]:
    normalized = normalize_vector(vector)
    z = GATE_BIAS
    for index, dim in enumerate(VECTOR_DIMS):
        z += GATE_WEIGHTS[dim] * normalized[index]
    g = 1 / (1 + math.exp(-z))
    if g >= 0.5:
        decision = "act"
    elif g >= 0.3:
        decision = "probe"
    else:
        decision = "replan"
    return g, decision


def gate_breakdown(vector: list[float]) -> list[dict[str, Any]]:
    normalized = normalize_vector(vector)
    rows = [
        {
            "dimension": dim,
            "value": normalized[index],
            "weight": GATE_WEIGHTS[dim],
            "contribution": GATE_WEIGHTS[dim] * normalized[index],
        }
        for index, dim in enumerate(VECTOR_DIMS)
    ]
    rows.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return rows


def empty_sigma(objective: str) -> Sigma:
    vector = [0.5] * len(VECTOR_DIMS)
    g, decision = gate(vector)
    return Sigma(
        objective=clip(objective, 4000),
        vector=vector,
        gate=g,
        decision=decision,
    )


def dedupe_latest(items: list[Any], limit: int, max_length: int) -> list[str]:
    seen: set[str] = set()
    collected: list[str] = []
    for item in reversed(items):
        value = clip(str(item).strip(), max_length)
        if not value or value in seen:
            continue
        seen.add(value)
        collected.append(value)
        if len(collected) >= limit:
            break
    collected.reverse()
    return collected


def maintain(sigma: Sigma) -> Sigma:
    preserved_goal = sigma.goal.copy()
    closed = len([item for item in sigma.subgoals if item.done])
    clean = strip_sensitive(sigma.to_dict())
    result = Sigma.from_dict(clean) if isinstance(clean, dict) else sigma.copy()
    result.goal = preserved_goal
    result.objective = clip(result.objective, 4000)
    seen_ids: set[str] = set()
    subgoals: list[Subgoal] = []
    for item in sigma.subgoals:
        if item.done:
            continue
        identifier = clip(item.id, 40)
        text = clip(item.text, 400)
        if not text:
            continue
        if identifier and identifier in seen_ids:
            continue
        if identifier:
            seen_ids.add(identifier)
        subgoals.append(Subgoal(identifier, text, False))
    result.subgoals = subgoals[:12]
    result.closed_subgoals = sigma.closed_subgoals + closed
    result.facts = dedupe_latest(result.facts, 20, 400)
    result.blockers = dedupe_latest(result.blockers, 8, 400)
    result.next_steps = [
        clip(str(item).strip(), 400) for item in result.next_steps if str(item).strip()
    ][:8]
    result.strategy = clip(str(result.strategy), 2000)
    needs_recompute = len(sigma.vector) != len(VECTOR_DIMS)
    result.vector = normalize_vector(result.vector)
    if needs_recompute:
        result.gate, result.decision = gate(result.vector)
    return result


def scan_balanced_json(source: str, expect: type | None = None) -> Any:
    for match in re.finditer(r"[{\[]", source):
        try:
            value, _ = _JSON_DECODER.raw_decode(source, match.start())
        except (json.JSONDecodeError, RecursionError):
            continue
        if expect is None or isinstance(value, expect):
            return value
    return None


def extract_json(text: str, expect: type | None = None) -> Any:
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        candidate = match.group(1)
        if candidate:
            parsed = scan_balanced_json(candidate, expect)
            if parsed is not None:
                return parsed
    return scan_balanced_json(text, expect)


def collect_strings(value: Any, limit: int) -> list[str]:
    if limit <= 0 or not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()][:limit]


async def call_think(
    think: Think,
    system: str,
    user: str,
    attempts: int = THINK_ATTEMPTS,
    base_delay: float = THINK_BASE_DELAY,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            result = await think(system, user)
        except AgentFault:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(base_delay * (2**attempt) + random.uniform(0.0, base_delay))
            continue
        if isinstance(result, str):
            return result
        return "" if result is None else str(result)
    raise AgentFault(
        "COGNITIVE_MODEL_UNAVAILABLE",
        f"A modellhívás ismételt próbálkozás után sem sikerült: {last_error}",
        502,
    )


async def think_json(
    think: Think,
    system: str,
    user: str,
    code: str,
    message: str,
    attempts: int = 2,
) -> dict[str, Any]:
    prompt = user
    for _ in range(max(1, attempts)):
        raw = await call_think(think, system, prompt)
        parsed = extract_json(raw, dict)
        if is_record(parsed):
            return parsed
        prompt = (
            f"{user}\n\nAz előző válaszod nem tartalmazott érvényes JSON objektumot. "
            "Adj vissza kizárólag egyetlen érvényes JSON objektumot, magyarázat és kódblokk nélkül."
            f"\n\nELŐZŐ VÁLASZ:\n{clip(raw, 2000)}"
        )
    raise AgentFault(code, message, 502)


async def deliberate(think: Think, sigma: Sigma, trace: str, guidance: str) -> Sigma:
    global _subgoal_counter
    system = "\n".join(
        [
            "Te a System 2 stratégiai tervező vagy. Kizárólag egyetlen JSON objektumot adj vissza, magyarázat nélkül.",
            'Séma: {"strategy": string, "subgoals": [{"id": string, "text": string, "done": boolean}], "next_steps": [string], "vector": [16 szám 0 és 1 között], "goal": {"state_pointers": [{"pointer": JSON-pointer a munkamemóriába, "equals": érték}], "required_files": [abszolút útvonal], "file_contents": [{"path": string, "contains": string}]}}',
            f"A vector dimenziói sorrendben: {', '.join(VECTOR_DIMS)}.",
            "A next_steps legfeljebb 5 konkrét, azonnal végrehajtható mikrolépés a System 1 számára.",
            "A goal csak olyan ellenőrizhető feltételeket tartalmazzon, amelyek a feladat valódi befejezését bizonyítják; ha nincs fájlkimenet, hagyd üresen a fájllistákat.",
            "Egyszerű beszélgetési üzenetnél vagy egylépéses kérdésnél a subgoals és a goal minden listája legyen üres.",
        ]
    )
    user = (
        f"MUNKAMEMÓRIA:\n{json.dumps(sigma.to_dict(), ensure_ascii=False)}"
        f"\n\nTANULT ÚTMUTATÁS:\n{guidance or 'nincs'}"
        f"\n\nFRISS NYOMVONAL:\n{clip_middle(trace, 8000, 0.2)}"
    )
    parsed = await think_json(
        think,
        system,
        user,
        "COGNITIVE_INVALID_JSON",
        "A stratégiai tervező (deliberate) érvénytelen JSON választ adott.",
    )
    result = sigma.copy()
    if isinstance(parsed.get("strategy"), str):
        result.strategy = parsed["strategy"]
    if isinstance(parsed.get("subgoals"), list):
        existing_by_id = {item.id: item for item in sigma.subgoals}
        existing_by_text = {item.text: item for item in sigma.subgoals}
        used_ids = {item.id for item in sigma.subgoals}
        taken: set[str] = set()
        collected: list[Subgoal] = []
        for entry in parsed["subgoals"]:
            if not is_record(entry):
                continue
            text = entry.get("text")
            text = clip(text.strip(), 400) if isinstance(text, str) else ""
            if not text:
                continue
            raw_id = entry.get("id")
            raw_id = clip(raw_id.strip(), 40) if isinstance(raw_id, str) else ""
            existing = existing_by_id.get(raw_id) if raw_id else None
            if existing is None:
                existing = existing_by_text.get(text)
            if existing is not None:
                if existing.id in taken:
                    continue
                taken.add(existing.id)
                collected.append(Subgoal(existing.id, text, existing.done))
                continue
            identifier = raw_id
            if not identifier or identifier in used_ids:
                while True:
                    async with _counter_lock:
                        _subgoal_counter += 1
                        identifier = f"sg_{_subgoal_counter:08x}"
                    if identifier not in used_ids:
                        break
            used_ids.add(identifier)
            collected.append(Subgoal(identifier, text, False))
        result.subgoals = collected
    if isinstance(parsed.get("next_steps"), list):
        result.next_steps = collect_strings(parsed["next_steps"], 5)
    vector = parsed.get("vector")
    if isinstance(vector, list) and len(vector) == len(VECTOR_DIMS):
        result.vector = normalize_vector(vector)
    goal = parsed.get("goal")
    if is_record(goal):
        pointers = goal.get("state_pointers")
        required_files = goal.get("required_files")
        file_contents = goal.get("file_contents")
        result.goal = GoalSpec(
            state_pointers=[
                {"pointer": item["pointer"], "equals": item["equals"]}
                for item in (pointers if isinstance(pointers, list) else [])
                if is_record(item)
                and isinstance(item.get("pointer"), str)
                and item["pointer"].startswith("/")
                and "equals" in item
            ][:10],
            required_files=list(
                dict.fromkeys(
                    path for path in collect_strings(required_files, 20) if path.startswith("/")
                )
            ),
            file_contents=[
                {"path": item["path"], "contains": item["contains"]}
                for item in (file_contents if isinstance(file_contents, list) else [])
                if is_record(item)
                and isinstance(item.get("path"), str)
                and item["path"].startswith("/")
                and isinstance(item.get("contains"), str)
                and item["contains"]
            ][:20],
        )
    g, decision = gate(result.vector)
    result.gate = g
    result.decision = decision
    return maintain(result)


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def apply_state_update(sigma: Sigma, args: dict[str, Any]) -> Sigma:
    result = sigma.copy()
    result.facts.extend(as_string_list(args.get("fact")))
    result.facts.extend(as_string_list(args.get("facts")))
    result.blockers.extend(as_string_list(args.get("blocker")))
    result.blockers.extend(as_string_list(args.get("blockers")))
    resolved = {
        item.lower()
        for item in (
            as_string_list(args.get("resolve_blocker"))
            + as_string_list(args.get("resolve_blockers"))
        )
    }
    if resolved:
        result.blockers = [
            item for item in result.blockers if item.strip().lower() not in resolved
        ]
    completed = as_string_list(args.get("complete_subgoal")) + as_string_list(
        args.get("complete_subgoals")
    )
    if completed:
        lowered = {item.lower() for item in completed}
        for subgoal in result.subgoals:
            if (
                subgoal.id in completed
                or subgoal.text in completed
                or subgoal.text.lower() in lowered
            ):
                subgoal.done = True
    steps = as_string_list(args.get("next_step")) + as_string_list(args.get("next_steps"))
    if steps:
        result.next_steps.extend(steps)
    return maintain(result)


def resolve_pointer(root: Any, path: str) -> Any:
    if path == "":
        return root
    if not path.startswith("/"):
        return _NOT_FOUND
    current = root
    for raw in path[1:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not re.match(r"^(0|[1-9][0-9]*)$", key):
                return _NOT_FOUND
            index = int(key)
            if index < 0 or index >= len(current):
                return _NOT_FOUND
            current = current[index]
        elif is_record(current):
            if key not in current:
                return _NOT_FOUND
            current = current[key]
        else:
            return _NOT_FOUND
    return current


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def deep_equal(left: Any, right: Any) -> bool:
    if is_number(left) and is_number(right):
        if isinstance(left, float) or isinstance(right, float):
            return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
        return left == right
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if left is None or right is None:
        return left is right
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(deep_equal(a, b) for a, b in zip(left, right))
    if is_record(left) and is_record(right):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(deep_equal(left[key], right[key]) for key in left)
    return False


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


async def safe_probe(probe: Probe, command: str) -> bool:
    try:
        return bool(await asyncio.wait_for(probe(command), timeout=PROBE_TIMEOUT_SECONDS))
    except Exception:
        return False


async def verify(sigma: Sigma, probe: Probe | None) -> dict[str, Any]:
    failures: list[str] = []
    passed: list[str] = []
    payload = sigma.to_dict()
    for pointer in sigma.goal.state_pointers:
        if "equals" not in pointer:
            failures.append(f"Hiányzó elvárt érték (equals): {pointer.get('pointer')}")
            continue
        label = str(pointer.get("pointer") or "")
        actual = resolve_pointer(payload, label)
        if actual is _NOT_FOUND or not deep_equal(actual, pointer.get("equals")):
            failures.append(f"Állapot nem egyezik: {pointer.get('pointer')}")
        else:
            passed.append(f"state:{label}")
    needs_files = bool(sigma.goal.required_files or sigma.goal.file_contents)
    if needs_files and probe is None:
        failures.append("Fájlellenőrzés szükséges, de nincs elérhető VM.")
    if probe is not None:
        checks: list[tuple[str, str, str]] = []
        for path in sigma.goal.required_files:
            checks.append(
                (f"Hiányzó fájl: {path}", f"file:{path}", f"test -f {shell_quote(path)}")
            )
        for entry in sigma.goal.file_contents:
            path = str(entry.get("path") or "")
            contains = str(entry.get("contains") or "")
            encoded = base64.b64encode(contains.encode("utf-8")).decode("ascii")
            command = (
                "node -e 'const fs=require(\"fs\");"
                "try{const c=fs.readFileSync(process.argv[1],\"utf8\");"
                "const n=Buffer.from(process.argv[2],\"base64\").toString(\"utf8\");"
                "process.exit(c.includes(n)?0:1);}catch{process.exit(1);}' "
                f"{shell_quote(path)} {shell_quote(encoded)}"
            )
            checks.append((f"Fájltartalom nem egyezik: {path}", f"content:{path}", command))
        outcomes = await asyncio.gather(*(safe_probe(probe, check[2]) for check in checks))
        for (failure, label, _), ok in zip(checks, outcomes):
            if ok:
                passed.append(label)
            else:
                failures.append(failure)
    if sigma.blockers:
        failures.append(f"Nyitott blokkolók: {'; '.join(sigma.blockers)}")
    open_subgoals = [item for item in sigma.subgoals if not item.done]
    if open_subgoals:
        failures.append(f"Lezáratlan részcélok: {', '.join(item.id for item in open_subgoals)}")
    has_evidence = bool(
        sigma.goal.state_pointers
        or sigma.goal.required_files
        or sigma.goal.file_contents
        or sigma.closed_subgoals > 0
        or sigma.facts
    )
    if not has_evidence:
        failures.append("Nincs ellenőrizhető bizonyíték a feladat teljesítésére.")
    return {"ok": not failures, "failures": failures, "passed": passed}


async def reflect(
    think: Think, objective: str, trace: str, failed_action: str
) -> Reflection:
    system = "\n".join(
        [
            "Hibaelemző vagy. A nyers nyomvonal alapján végezz gyökérok-elemzést. Kizárólag JSON-t adj vissza.",
            'Séma: {"failure_point": a kudarc pontos lépése, "component": felelős komponens (planner|executor|tool|environment|verifier|model), "root_cause": string, "pivot_action": egyetlen konkrét korrekciós lépés, "patch": javítócsomag leírása, "corrective_action": a sikertelen műveletet közvetlenül kiváltó helyes parancs vagy művelet}',
        ]
    )
    user = (
        f"CÉL:\n{clip(objective, 4000)}"
        f"\n\nSIKERTELEN MŰVELET:\n{clip(failed_action, 2000)}"
        f"\n\nNYOMVONAL:\n{clip_middle(trace, 10000, 0.2)}"
    )
    parsed = await think_json(
        think,
        system,
        user,
        "COGNITIVE_INVALID_JSON",
        "A hibaelemző (reflect) érvénytelen JSON választ adott.",
    )

    def required_text(key: str) -> str:
        value = parsed.get(key)
        if not isinstance(value, str) or not value.strip():
            raise AgentFault(
                "COGNITIVE_MISSING_FIELD",
                f"A hibaelemzésből hiányzik a kötelező mező: {key}",
                502,
            )
        return clip(value.strip(), 2000)

    def optional_text(key: str) -> str:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return clip(value.strip(), 2000)
        return ""

    component = required_text("component").lower()
    if component not in ALLOWED_COMPONENTS:
        raise AgentFault(
            "COGNITIVE_INVALID_COMPONENT",
            f"Érvénytelen komponens a hibaelemzésben: {component}",
            502,
        )
    pivot_action = required_text("pivot_action")
    patch = optional_text("patch")
    corrective_action = optional_text("corrective_action") or patch or pivot_action
    return Reflection(
        failure_point=required_text("failure_point"),
        component=component,
        root_cause=required_text("root_cause"),
        pivot_action=pivot_action,
        patch=patch,
        failed_action=clip(failed_action, 2000),
        corrective_action=corrective_action,
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+|\d+|_+", text.lower(), re.UNICODE)


def fnv_hash(text: str) -> int:
    value = 0x811C9DC5
    for char in text:
        value ^= ord(char)
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


@lru_cache(maxsize=4096)
def embed_cached(text: str) -> tuple[float, ...]:
    vector = [0.0] * EMBED_DIM
    tokens = tokenize(text)
    for index, token in enumerate(tokens):
        position = fnv_hash(token) % EMBED_DIM
        vector[position] += 1
        if index + 1 < len(tokens):
            next_position = fnv_hash(f"{token} {tokens[index + 1]}") % EMBED_DIM
            vector[next_position] += 0.5
        for offset in range(len(token) - 2):
            trigram = f"#{token[offset : offset + 3]}"
            trigram_position = fnv_hash(trigram) % EMBED_DIM
            vector[trigram_position] += 0.25
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return tuple(value / norm for value in vector)


def embed(text: str) -> list[float]:
    return list(embed_cached(text))


def parse_vector(raw: Any) -> list[float]:
    if isinstance(raw, list):
        result: list[float] = []
        for item in raw:
            if isinstance(item, bool):
                result.append(0.0)
            elif isinstance(item, (int, float)):
                result.append(float(item) if math.isfinite(float(item)) else 0.0)
            elif isinstance(item, str):
                try:
                    number = float(item)
                except ValueError:
                    result.append(0.0)
                else:
                    result.append(number if math.isfinite(number) else 0.0)
            else:
                result.append(0.0)
        return result
    if isinstance(raw, str):
        trimmed = raw.strip()
        if not trimmed:
            return []
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError:
            inner = trimmed.strip("[](){}")
            if not inner:
                return []
            values: list[float] = []
            for part in inner.split(","):
                try:
                    number = float(part.strip())
                except ValueError:
                    values.append(0.0)
                else:
                    values.append(number if math.isfinite(number) else 0.0)
            return values
        if isinstance(parsed, list):
            return parse_vector(parsed)
        return []
    return []


def cosine(left: Any, right: Any) -> float:
    a = parse_vector(left)
    b = parse_vector(right)
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for index in range(len(a)):
        x = a[index]
        y = b[index]
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denominator = math.sqrt(norm_a) * math.sqrt(norm_b)
    return dot / denominator if denominator > 0 else 0.0


def wilson_interval(successes: int, failures: int, z: float = 1.96) -> tuple[float, float]:
    trials = successes + failures
    if trials <= 0:
        return 0.0, 1.0
    proportion = successes / trials
    denominator = 1 + (z * z) / trials
    centre = proportion + (z * z) / (2 * trials)
    margin = z * math.sqrt((proportion * (1 - proportion) + (z * z) / (4 * trials)) / trials)
    return (
        clamp((centre - margin) / denominator, 0.0, 1.0),
        clamp((centre + margin) / denominator, 0.0, 1.0),
    )


async def retrieve(objective: str, context: str) -> dict[str, Any]:
    database = database_module.get_admin_database()
    skills_response, playbooks_response = await asyncio.gather(
        database.table("agent_skills")
        .select("id", "name", "dsl", "terms", "vector", "successes", "failures")
        .eq("status", "active")
        .limit(500)
        .execute(),
        database.table("agent_playbooks")
        .select("content")
        .order("created_at", desc=True)
        .limit(1)
        .execute(),
    )
    skills = skills_response.rows
    playbooks = playbooks_response.rows
    query = f"{objective}\n{context}"
    query_vector = embed(query)
    query_tokens = set(tokenize(query))
    ranked: list[tuple[Skill, float]] = []
    for row in skills:
        vector = parse_vector(row.get("vector"))
        terms = set(tokenize(f"{row.get('name') or ''} {row.get('terms') or ''}"))
        if terms:
            matched = len([term for term in terms if term in query_tokens])
            lexical = matched / math.sqrt(len(terms) * max(1, len(query_tokens)))
        else:
            lexical = 0.0
        successes = max(0, to_int(row.get("successes")))
        failures = max(0, to_int(row.get("failures")))
        rate = (successes + 1) / (successes + failures + 2)
        score = 0.5 * cosine(query_vector, vector) + 0.3 * min(1.0, lexical) + 0.2 * rate
        if score > 0.25:
            skill = Skill(
                id=str(row.get("id")),
                name=str(row.get("name") or ""),
                dsl=str(row.get("dsl") or ""),
                terms=str(row.get("terms") or ""),
                vector=vector,
                successes=successes,
                failures=failures,
            )
            ranked.append((skill, score))
    ranked.sort(key=lambda entry: entry[1], reverse=True)
    pool = list(ranked)
    selected: list[tuple[Skill, float]] = []
    while pool and len(selected) < 3:
        best_index = 0
        best_value = -math.inf
        for position, (candidate, score) in enumerate(pool):
            redundancy = max(
                (cosine(candidate.vector, chosen.vector) for chosen, _ in selected),
                default=0.0,
            )
            value = 0.7 * score - 0.3 * redundancy
            if value > best_value:
                best_value = value
                best_index = position
        selected.append(pool.pop(best_index))
    playbook = str(playbooks[0].get("content") or "") if playbooks else ""
    return {
        "skills": [skill for skill, _ in selected],
        "playbook": playbook,
        "scores": {skill.id: round(score, 4) for skill, score in selected},
    }


def ngrams(text: str) -> list[str]:
    tokens = tokenize(text)
    return [f"{tokens[index]} {tokens[index + 1]}" for index in range(len(tokens) - 1)]


def distribution(grams: list[str]) -> dict[str, float]:
    if not grams:
        return {}
    counts: dict[str, int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    return {key: value / len(grams) for key, value in counts.items()}


async def apply_policy_delta(ngram: str, delta: float) -> None:
    database = database_module.get_admin_database()
    for _ in range(8):
        current = await (
            database.table("agent_policy_weights")
            .select("ngram", "weight", "observations")
            .eq("ngram", ngram)
            .limit(1)
            .execute()
        )
        rows = current.rows
        now = datetime.now(UTC).isoformat()
        if not rows:
            try:
                await (
                    database.table("agent_policy_weights")
                    .insert(
                        {
                            "ngram": ngram,
                            "weight": clamp(delta, -POLICY_WEIGHT_BOUND, POLICY_WEIGHT_BOUND),
                            "observations": 1,
                            "updated_at": now,
                        }
                    )
                    .execute()
                )
                return
            except Exception as exc:
                if getattr(exc, "code", None) == "23505":
                    continue
                raise
        row = rows[0]
        observations = to_int(row.get("observations"))
        weight = to_float(row.get("weight"))
        new_weight = clamp(weight * POLICY_DECAY + delta, -POLICY_WEIGHT_BOUND, POLICY_WEIGHT_BOUND)
        updated = await (
            database.table("agent_policy_weights")
            .update({"weight": new_weight, "observations": observations + 1, "updated_at": now})
            .eq("ngram", ngram)
            .eq("observations", observations)
            .execute()
        )
        if updated.rows:
            return
    raise AgentFault(
        "COGNITIVE_POLICY_CONFLICT",
        f"A politikasúly frissítése ütközés miatt sikertelen: {ngram}",
        500,
    )


async def distill(pairs: list[dict[str, Any]]) -> None:
    deltas: dict[str, float] = {}
    epsilon = 1e-3
    learning_rate = 0.5
    for pair in pairs:
        failed = ngrams(str(pair.get("failed") or ""))
        corrective = ngrams(str(pair.get("corrective") or ""))
        if not failed or not corrective:
            continue
        advantage = pair.get("advantage")
        if not isinstance(advantage, (int, float)) or isinstance(advantage, bool) or not math.isfinite(advantage):
            advantage = 0.0
        advantage = min(1.0, max(0.0, float(advantage)))
        if advantage <= 0:
            continue
        p_distribution = distribution(failed)
        q_distribution = distribution(corrective)
        for gram in set(p_distribution) | set(q_distribution):
            q_value = q_distribution.get(gram, 0.0) + epsilon
            p_value = p_distribution.get(gram, 0.0) + epsilon
            delta = learning_rate * advantage * (q_value + p_value) * math.log(q_value / p_value)
            if abs(delta) > 1e-4:
                deltas[gram] = deltas.get(gram, 0.0) + delta
    if not deltas:
        return
    selected = sorted(deltas.items(), key=lambda entry: abs(entry[1]), reverse=True)[:400]
    async with _db_write_lock:
        batch_size = 25
        for offset in range(0, len(selected), batch_size):
            batch = selected[offset : offset + batch_size]
            await asyncio.gather(
                *(apply_policy_delta(gram, delta) for gram, delta in batch)
            )


async def policy_guidance() -> str:
    database = database_module.get_admin_database()
    favor_response, avoid_response = await asyncio.gather(
        database.table("agent_policy_weights")
        .select("ngram", "weight")
        .gt("weight", 0.01)
        .order("weight", desc=True)
        .limit(8)
        .execute(),
        database.table("agent_policy_weights")
        .select("ngram", "weight")
        .lt("weight", -0.01)
        .order("weight", desc=False)
        .limit(8)
        .execute(),
    )
    lines: list[str] = []
    favor = favor_response.rows
    avoid = avoid_response.rows
    if favor:
        lines.append("FAVOR: " + ", ".join(f'"{row.get("ngram")}"' for row in favor))
    if avoid:
        lines.append("AVOID: " + ", ".join(f'"{row.get("ngram")}"' for row in avoid))
    return "\n".join(lines)


def valid_dsl(dsl: str) -> bool:
    lines = [line.strip() for line in dsl.split("\n") if line.strip()]
    if not lines:
        return False

    def has(keyword: str) -> bool:
        return any(
            line.startswith(f"{keyword} ") or line.startswith(f"{keyword}:") for line in lines
        )

    first = lines[0]
    if not (first.startswith("SKILL ") or first.startswith("SKILL:")):
        return False
    return all(has(keyword) for keyword in DSL_KEYWORDS)


def skill_solves(skill_tokens: set[str], skill_vector: list[float], task: dict[str, Any]) -> bool:
    needed = list(
        {
            token
            for token in tokenize(
                f"{task.get('pivot_action') or ''} {task.get('root_cause') or ''}"
            )
            if len(token) >= 4
        }
    )
    if not needed:
        return False
    covered = len([token for token in needed if token in skill_tokens]) / len(needed)
    if covered < 0.5:
        return False
    objective = f"{task.get('objective') or ''}\n{task.get('root_cause') or ''}"
    return cosine(embed(objective), skill_vector) >= 0.2


async def regression_gate(
    candidate: str, cluster_reflection_ids: set[str], cluster_task_ids: set[str], component: str
) -> dict[str, Any]:
    database = database_module.get_admin_database()
    active_response, reflection_response = await asyncio.gather(
        database.table("agent_skills").select("dsl").eq("status", "active").limit(100).execute(),
        database.table("agent_reflections")
        .select("id", "task_id", "objective", "component", "root_cause", "pivot_action")
        .eq("component", component)
        .order("created_at", desc=True)
        .limit(200)
        .execute(),
    )
    seen_objectives: set[str] = set()
    held_out: list[dict[str, Any]] = []
    for row in reflection_response.rows:
        if str(row.get("id")) in cluster_reflection_ids or str(row.get("task_id")) in cluster_task_ids:
            continue
        objective = str(row.get("objective") or "")
        if objective in seen_objectives:
            continue
        seen_objectives.add(objective)
        held_out.append(row)
        if len(held_out) >= 20:
            break
    if not held_out:
        return {"accept": False, "baseline": 0.0, "candidate": 0.0}
    base_skills = [
        (set(tokenize(str(row.get("dsl") or ""))), embed(str(row.get("dsl") or "")))
        for row in active_response.rows
    ]
    candidate_tokens = set(tokenize(candidate))
    candidate_vector = embed(candidate)
    baseline_solved = 0
    candidate_solved = 0
    for task in held_out:
        base_hit = any(
            skill_solves(tokens, vector, task) for tokens, vector in base_skills
        )
        candidate_hit = base_hit or skill_solves(candidate_tokens, candidate_vector, task)
        if base_hit:
            baseline_solved += 1
        if candidate_hit:
            candidate_solved += 1
    baseline = baseline_solved / len(held_out)
    candidate_rate = candidate_solved / len(held_out)
    return {
        "accept": candidate_rate > baseline,
        "baseline": baseline,
        "candidate": candidate_rate,
    }


async def synthesize_skills(think: Think, log: Callable[[str], Awaitable[None]]) -> None:
    database = database_module.get_admin_database()
    reflection_response, skills_response = await asyncio.gather(
        database.table("agent_reflections")
        .select("id", "task_id", "component", "root_cause", "pivot_action", "patch", "objective")
        .order("created_at", desc=True)
        .limit(50)
        .execute(),
        database.table("agent_skills").select("vector").limit(1000).execute(),
    )
    items = [
        {
            **row,
            "v": embed(f"{row.get('component', '')} {row.get('root_cause', '')}"),
        }
        for row in reflection_response.rows
    ]
    known_vectors: list[Any] = [row.get("vector") for row in skills_response.rows]
    used: set[int] = set()
    for index in range(len(items)):
        if index in used:
            continue
        seed = items[index]
        cluster = [index]
        for other_index in range(index + 1, len(items)):
            if other_index in used:
                continue
            other = items[other_index]
            if other.get("component") != seed.get("component"):
                continue
            if cosine(seed["v"], other["v"]) >= 0.55:
                cluster.append(other_index)
        if len(cluster) < 2:
            continue
        used.update(cluster)
        group = [items[position] for position in cluster]
        centroid = embed(
            " ".join(f"{item.get('component', '')} {item.get('root_cause', '')}" for item in group)
        )
        if any(cosine(vector, centroid) >= 0.8 for vector in known_vectors):
            continue
        system = "\n".join(
            [
                "Meta-ágens vagy: a visszatérő hibamintából újrahasznosítható képességet szintetizálsz SKILL DSL nyelven. Csak a DSL-t add vissza.",
                "Kötelező formátum, minden záradék külön sorban:",
                "SKILL <rövid_snake_case_név>",
                "WHEN <alkalmazási feltétel>",
                "REQUIRE <előfeltétel>",
                "STEP <lépés> (több STEP sor is lehet)",
                "VERIFY <ellenőrizhető sikerfeltétel>",
                "RECOVER <helyreállítási lépés hiba esetén>",
            ]
        )
        user_prompt = "\n---\n".join(
            f"KOMPONENS: {item.get('component')}\n"
            f"GYÖKÉROK: {item.get('root_cause')}\n"
            f"KORREKCIÓ: {item.get('pivot_action')}\n"
            f"JAVÍTÁS: {item.get('patch')}"
            for item in group
        )
        dsl = re.sub(r"```[a-z]*\n?|```", "", await call_think(think, system, user_prompt)).strip()
        if not valid_dsl(dsl):
            await log("A szintetizált képesség érvénytelen DSL-t adott, elvetve.")
            continue
        name_match = re.search(r"^[ \t]*SKILL[: \t]+(\S[^\n]*)$", dsl, re.MULTILINE)
        name = (
            name_match.group(1).strip()[:80]
            if name_match
            else f"skill_{hashlib.sha256(dsl.encode('utf-8')).hexdigest()[:10]}"
        )
        gate_result = await regression_gate(
            dsl,
            {str(item.get("id")) for item in group},
            {str(item.get("task_id")) for item in group},
            str(seed.get("component")),
        )
        await (
            database.table("agent_skills")
            .insert(
                {
                    "name": name,
                    "dsl": dsl,
                    "terms": " ".join(tokenize(dsl)[:200]),
                    "vector": centroid,
                    "status": "active" if gate_result["accept"] else "rejected",
                    "baseline_rate": gate_result["baseline"],
                    "candidate_rate": gate_result["candidate"],
                }
            )
            .execute()
        )
        known_vectors.append(centroid)
        await log(
            f"Új képesség „{name}”: "
            f"{'aktiválva' if gate_result['accept'] else 'elutasítva'} "
            f"(alap {gate_result['baseline']:.2f}, jelölt {gate_result['candidate']:.2f})."
        )


async def prune_skills(log: Callable[[str], Awaitable[None]]) -> None:
    database = database_module.get_admin_database()
    response = await (
        database.table("agent_skills")
        .select("id", "name", "successes", "failures")
        .eq("status", "active")
        .limit(500)
        .execute()
    )
    for row in response.rows:
        successes = max(0, to_int(row.get("successes")))
        failures = max(0, to_int(row.get("failures")))
        if successes + failures < SKILL_RETIRE_MIN_TRIALS:
            continue
        _, upper = wilson_interval(successes, failures)
        if upper >= SKILL_RETIRE_UPPER_BOUND:
            continue
        updated = await (
            database.table("agent_skills")
            .update({"status": "rejected"})
            .eq("id", str(row.get("id")))
            .eq("status", "active")
            .execute()
        )
        if updated.rows:
            await log(
                f"A képesség visszavonva a tartósan alacsony sikerarány miatt: {row.get('name') or row.get('id')}."
            )


async def evolve_playbook(think: Think, log: Callable[[str], Awaitable[None]]) -> None:
    database = database_module.get_admin_database()
    tasks_query = database.table("agent_tasks").select("id", count="exact", head=True)
    count_method = getattr(tasks_query, "count_exact", None)
    if callable(count_method):
        count_res = await count_method()
        total = to_int(getattr(count_res, "count", count_res))
    else:
        tasks_response = await tasks_query.execute()
        total = to_int(getattr(tasks_response, "count", len(getattr(tasks_response, "rows", []))))
    last_response = await (
        database.table("agent_playbooks")
        .select("task_count")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    last_count = 0
    if last_response.rows:
        last_count = to_int(last_response.rows[0].get("task_count"))
    if total - last_count < 5:
        return
    reflection_response, skills_response, previous_response = await asyncio.gather(
        database.table("agent_reflections")
        .select("component", "root_cause", "pivot_action")
        .order("created_at", desc=True)
        .limit(40)
        .execute(),
        database.table("agent_skills").select("name", "dsl").eq("status", "active").limit(20).execute(),
        database.table("agent_playbooks")
        .select("content")
        .order("created_at", desc=True)
        .limit(1)
        .execute(),
    )
    system = (
        "Tapasztalati tudáskonszolidátor vagy. Írj tömör, ember által olvasható magyar működési "
        "kézikönyvet két szakasszal: 'Bevált eljárások' és 'Elkerülendő anti-patternek'. "
        "Legfeljebb 40 pont."
    )
    reflections = "\n".join(
        f"- [{row.get('component')}] {row.get('root_cause')} => {row.get('pivot_action')}"
        for row in reflection_response.rows
    )
    active_skills = "\n\n".join(str(row.get("dsl") or "") for row in skills_response.rows)
    previous_content = (
        str(previous_response.rows[0].get("content") or "nincs") if previous_response.rows else "nincs"
    )
    content = (
        await call_think(
            think,
            system,
            f"ELŐZŐ KÉZIKÖNYV:\n{previous_content}\n\nREFLEXIÓK:\n{reflections}\n\nAKTÍV KÉPESSÉGEK:\n{active_skills}",
        )
    ).strip()
    if not content:
        return
    if content == previous_content.strip():
        return
    inserted = await (
        database.table("agent_playbooks")
        .insert({"content": clip(content, 20000), "task_count": total})
        .execute()
    )
    if not inserted.rows:
        return
    own_id = str(inserted.rows[0].get("id"))
    same_count = await (
        database.table("agent_playbooks")
        .select("id")
        .eq("task_count", total)
        .order("created_at", desc=False)
        .order("id", desc=False)
        .execute()
    )
    if same_count.rows and str(same_count.rows[0].get("id")) != own_id:
        await database.table("agent_playbooks").delete().eq("id", own_id).execute()
        return
    await log("A működési kézikönyv frissült.")


async def record_skill_outcome(skill_id: str, verified: bool) -> None:
    database = database_module.get_admin_database()
    for _ in range(8):
        current = await (
            database.table("agent_skills")
            .select("successes", "failures")
            .eq("id", skill_id)
            .limit(1)
            .execute()
        )
        if not current.rows:
            return
        row = current.rows[0]
        successes = to_int(row.get("successes"))
        failures = to_int(row.get("failures"))
        patch = (
            {"successes": successes + 1} if verified else {"failures": failures + 1}
        )
        updated = await (
            database.table("agent_skills")
            .update(patch)
            .eq("id", skill_id)
            .eq("successes", successes)
            .eq("failures", failures)
            .execute()
        )
        if updated.rows:
            return
    raise AgentFault(
        "COGNITIVE_SKILL_CONFLICT",
        f"A képesség eredményének rögzítése ütközés miatt sikertelen: {skill_id}",
        500,
    )


def normalize_reflection(item: Any) -> dict[str, str] | None:
    if isinstance(item, Reflection):
        source = asdict(item)
    elif isinstance(item, dict):
        source = item
    else:
        return None
    return {key: clip(str(source.get(key) or ""), 2000) for key in REFLECTION_FIELDS}


async def consolidate(
    think: Think,
    payload: dict[str, Any],
    log: Callable[[str], Awaitable[None]],
) -> None:
    database = database_module.get_admin_database()
    task_id = str(payload.get("taskId") or "")
    objective = str(payload.get("objective") or "")
    outcome = str(payload.get("outcome") or "")
    steps = max(0, to_int(payload.get("steps")))
    verified = bool(payload.get("verified"))
    raw_reflections = payload.get("reflections")
    reflections = [
        normalized
        for normalized in (
            normalize_reflection(item)
            for item in (raw_reflections if isinstance(raw_reflections, list) else [])
        )
        if normalized is not None
    ]
    raw_skills = payload.get("usedSkills")
    used_skills = raw_skills if isinstance(raw_skills, list) else []
    await (
        database.table("agent_tasks")
        .upsert(
            {
                "id": task_id,
                "objective": clip(objective, 4000),
                "outcome": outcome,
                "steps": steps,
                "verified": verified,
            },
            on_conflict="id",
        )
        .execute()
    )
    if reflections:
        rows = [
            {
                "task_id": task_id,
                "objective": clip(objective, 4000),
                "failure_point": reflection["failure_point"],
                "component": reflection["component"],
                "root_cause": reflection["root_cause"],
                "pivot_action": reflection["pivot_action"],
                "patch": reflection["patch"],
                "failed_action": reflection["failed_action"],
                "corrective_action": reflection["corrective_action"],
            }
            for reflection in reflections
        ]
        await database.table("agent_reflections").insert(rows).execute()
        await distill(
            [
                {
                    "failed": reflection["failed_action"],
                    "corrective": reflection["corrective_action"],
                    "advantage": 1 if verified else 0.25,
                }
                for reflection in reflections
                if reflection["corrective_action"]
            ]
        )
    for skill_id in dict.fromkeys(str(item) for item in used_skills):
        await record_skill_outcome(skill_id, verified)
    async with _db_write_lock:
        errors: list[Exception] = []
        if reflections:
            try:
                await synthesize_skills(think, log)
            except Exception as exc:
                errors.append(exc)
        try:
            await prune_skills(log)
        except Exception as exc:
            errors.append(exc)
        try:
            await evolve_playbook(think, log)
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise errors[0]


async def take_operator_messages(task_id: str) -> list[str]:
    database = database_module.get_admin_database()
    response = await (
        database.table("agent_operator_messages")
        .update({"consumed": True})
        .eq("task_id", task_id)
        .eq("consumed", False)
        .execute()
    )
    rows = sorted(response.rows, key=lambda row: str(row.get("created_at") or ""))
    return [str(row.get("content") or "") for row in rows]


async def post_operator_message(task_id: str, content: str) -> None:
    if not content or not content.strip():
        raise AgentFault("INVALID_ARGUMENT", "Az operátori üzenet nem lehet üres.", 400)
    database = database_module.get_admin_database()
    await (
        database.table("agent_operator_messages")
        .insert({"task_id": task_id, "content": clip(content.strip(), 8000)})
        .execute()
    )


async def reason_node(
    think: Think,
    question: str,
    depth: int,
    branches: int,
    cache: dict[tuple[str, int], "asyncio.Future[ReasonNode]"] | None = None,
) -> ReasonNode:
    if cache is None:
        cache = {}
    key = (question, depth)
    existing = cache.get(key)
    if existing is not None:
        return await existing
    task = asyncio.ensure_future(expand_reason_node(think, question, depth, branches, cache))
    cache[key] = task
    return await task


async def expand_reason_node(
    think: Think,
    question: str,
    depth: int,
    branches: int,
    cache: dict[tuple[str, int], "asyncio.Future[ReasonNode]"],
) -> ReasonNode:
    if depth <= 0:
        answer = await call_think(think, ATOMIC_REASONER_SYSTEM, question)
        return ReasonNode(question=question, answer=clip(answer, 3000), children=[])
    decomposition = await call_think(
        think,
        f"Bontsd a kérdést legfeljebb {branches} független, kisebb alkérdésre. "
        "Csak JSON tömböt adj vissza stringekkel. Ha a kérdés atomi, adj vissza üres tömböt.",
        question,
    )
    sub_questions = list(
        dict.fromkeys(collect_strings(extract_json(decomposition, list), branches))
    )
    children = list(
        await asyncio.gather(
            *(
                reason_node(think, sub_question, depth - 1, branches, cache)
                for sub_question in sub_questions
            )
        )
    )
    if children:
        answer = await call_think(
            think,
            "Összegző vagy. Az alkérdésekre adott válaszokból adj tömör, következetes választ az eredeti kérdésre.",
            "EREDETI KÉRDÉS:\n"
            f"{question}\n\nALVÁLASZOK:\n"
            + "\n".join(f"- {child.question}: {child.answer}" for child in children),
        )
    else:
        answer = await call_think(think, ATOMIC_REASONER_SYSTEM, question)
    return ReasonNode(question=question, answer=clip(answer, 3000), children=children)


def reason_node_to_dict(node: ReasonNode) -> dict[str, Any]:
    return {
        "question": node.question,
        "answer": node.answer,
        "children": [reason_node_to_dict(child) for child in node.children],
    }


def flatten_reason_tree(node: ReasonNode) -> list[ReasonNode]:
    collected = [node]
    for child in node.children:
        collected.extend(flatten_reason_tree(child))
    return collected


def reason_tree_stats(node: ReasonNode) -> dict[str, int]:
    def measure(current: ReasonNode) -> int:
        if not current.children:
            return 0
        return 1 + max(measure(child) for child in current.children)

    flat = flatten_reason_tree(node)
    return {
        "nodes": len(flat),
        "leaves": len([item for item in flat if not item.children]),
        "depth": measure(node),
    }


async def reason_tree(think: Think, question: str, depth: int, branches: int) -> ReasonNode:
    if not isinstance(question, str) or not question.strip():
        raise AgentFault("INVALID_ARGUMENT", "A kérdés nem lehet üres.", 400)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0 or depth > REASON_MAX_DEPTH:
        raise AgentFault(
            "INVALID_ARGUMENT",
            f"Érvénytelen mélység: 0 és {REASON_MAX_DEPTH} közötti egész szám szükséges.",
            400,
        )
    if (
        isinstance(branches, bool)
        or not isinstance(branches, int)
        or branches < 1
        or branches > REASON_MAX_BRANCHES
    ):
        raise AgentFault(
            "INVALID_ARGUMENT",
            f"Érvénytelen elágazásszám: 1 és {REASON_MAX_BRANCHES} közötti egész szám szükséges.",
            400,
        )
    nodes = 0
    level = 1
    for _ in range(depth + 1):
        nodes += level
        level *= branches
    if nodes * 2 > REASON_MAX_CALLS:
        raise AgentFault(
            "INVALID_ARGUMENT",
            f"A kért fa legfeljebb {nodes * 2} modellhívást igényelhet, a korlát {REASON_MAX_CALLS}.",
            400,
        )
    cache: dict[tuple[str, int], "asyncio.Future[ReasonNode]"] = {}
    try:
        return await reason_node(think, question.strip(), depth, branches, cache)
    except BaseException:
        for task in cache.values():
            task.cancel()
        raise


def sigma_fingerprint(sigma: Sigma) -> str:
    canonical = json.dumps(
        sigma.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sigma_diff(before: Sigma, after: Sigma) -> dict[str, Any]:
    before_facts = set(before.facts)
    after_facts = set(after.facts)
    before_blockers = set(before.blockers)
    after_blockers = set(after.blockers)
    before_ids = {item.id for item in before.subgoals}
    after_ids = {item.id for item in after.subgoals}
    return {
        "facts_added": [item for item in after.facts if item not in before_facts],
        "facts_removed": [item for item in before.facts if item not in after_facts],
        "blockers_added": [item for item in after.blockers if item not in before_blockers],
        "blockers_resolved": [item for item in before.blockers if item not in after_blockers],
        "subgoals_added": [item.id for item in after.subgoals if item.id not in before_ids],
        "subgoals_removed": [item.id for item in before.subgoals if item.id not in after_ids],
        "closed_delta": after.closed_subgoals - before.closed_subgoals,
        "gate_delta": after.gate - before.gate,
        "decision_changed": before.decision != after.decision,
    }


def completion_ratio(sigma: Sigma) -> float:
    total = sigma.closed_subgoals + len(sigma.subgoals)
    if total <= 0:
        return 0.0
    done = len([item for item in sigma.subgoals if item.done])
    return clamp((sigma.closed_subgoals + done) / total, 0.0, 1.0)


def detect_loop(actions: list[str], window: int = 4) -> bool:
    normalized = [re.sub(r"\s+", " ", item.strip().lower()) for item in actions]
    if window < 2 or len(normalized) < window:
        return False
    recent = normalized[-window:]
    if len(set(recent)) == 1:
        return True
    if window >= 4 and window % 2 == 0 and recent[0] != recent[1]:
        return all(recent[index] == recent[index % 2] for index in range(window))
    return False


class Lifecycle:
    def __init__(self, on_change: Callable[[str, str], Awaitable[None]]):
        self.phase = "osPerceive"
        self.trail: list[str] = ["osPerceive"]
        self.timeline: list[dict[str, Any]] = [
            {"phase": "osPerceive", "entered_at": time.monotonic()}
        ]
        self._on_change = on_change
        self._lock = asyncio.Lock()

    @property
    def terminal(self) -> bool:
        return not TRANSITIONS.get(self.phase, ())

    @property
    def allowed(self) -> tuple[str, ...]:
        return TRANSITIONS.get(self.phase, ())

    def can_transition(self, next_phase: str) -> bool:
        return next_phase in TRANSITIONS.get(self.phase, ())

    def visits(self, phase: str) -> int:
        return len([item for item in self.trail if item == phase])

    def durations(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        now = time.monotonic()
        for index, entry in enumerate(self.timeline):
            end = self.timeline[index + 1]["entered_at"] if index + 1 < len(self.timeline) else now
            totals[entry["phase"]] = totals.get(entry["phase"], 0.0) + (end - entry["entered_at"])
        return totals

    async def to(self, next_phase: str) -> None:
        async with self._lock:
            if next_phase not in TRANSITIONS.get(self.phase, ()):
                raise AgentFault(
                    "INVALID_PHASE_TRANSITION",
                    f"Tiltott fázisátmenet: {self.phase} -> {next_phase}",
                    400,
                )
            previous = self.phase
            await self._on_change(previous, next_phase)
            self.phase = next_phase
            self.trail.append(next_phase)
            self.timeline.append({"phase": next_phase, "entered_at": time.monotonic()})