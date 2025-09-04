# adaptive_engine.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Literal, Union
from math import exp

# ============================
# Tipos / Modelos (Pydantic v2)
# ============================

DifficultyLevel = Literal["easy", "medium", "hard"]


class DifficultyTag(BaseModel):
    level: DifficultyLevel
    numeric: Literal[1, 2, 3]


class ItemIRT(BaseModel):
    a: float = 1.0  # discriminación
    b: float = 0.0  # dificultad
    c: float = 0.2  # azar


class HintML(BaseModel):
    tier: int
    text: Dict[str, str]


class MCQOption(BaseModel):
    id: str
    text: Dict[str, str]


class MCQAnswerKey(BaseModel):
    options: List[MCQOption]
    correct_option_id: str


class ItemBase(BaseModel):
    item_id: str
    type: Literal["numeric", "multiple_choice"]
    skills: List[str]
    difficulty: DifficultyTag
    time_limit_sec: int = 60
    irt_params: Optional[ItemIRT] = None
    content_refs: Optional[List[str]] = None


class NumericItem(ItemBase):
    type: Literal["numeric"] = "numeric"
    stem: Dict[str, str]
    answer_key: Dict[str, float]  # {"value": 5, "tolerance": 0}
    hints: Optional[List[HintML]] = None


class MCQItem(ItemBase):
    type: Literal["multiple_choice"] = "multiple_choice"
    stem: Dict[str, str]
    answer_key: MCQAnswerKey
    hints: Optional[List[HintML]] = None


Item = Union[NumericItem, MCQItem]


class RetryPolicy(BaseModel):
    max_attempts_per_item: int = 2
    cooldown_sec: int = 0


class DeliveryConstraints(BaseModel):
    max_items_this_session: int = 6
    target_mastery: float = 0.8
    path_policy: Literal["mastery_first"] = "mastery_first"
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    escalation_rules: List[str] = Field(default_factory=list)


class RecentPerf(BaseModel):
    accuracy_last_n: Optional[float] = None
    avg_response_time_sec: Optional[int] = None
    streak_correct: Optional[int] = 0
    streak_incorrect: Optional[int] = 0


class StudentState(BaseModel):
    estimated_level: Optional[int] = 1  # 1..3
    recent_performance: RecentPerf = Field(default_factory=RecentPerf)
    known_gaps: List[str] = Field(default_factory=list)
    confidence_model_enabled: Optional[bool] = True


class AdaptivePolicies(BaseModel):
    mastery_estimator: Literal["rolling_accuracy_weighted_time"] = "rolling_accuracy_weighted_time"
    irt_routing: bool = True
    use_known_gaps_biasing: bool = True


class PackageLike(BaseModel):
    session_id: str
    locale: str = "es-EC"
    available_locales: List[str] = Field(default_factory=lambda: ["es-EC"])
    items: List[Item]
    delivery_constraints: DeliveryConstraints
    student_state: StudentState = Field(default_factory=StudentState)
    adaptive_policies: AdaptivePolicies = Field(default_factory=AdaptivePolicies)

    @field_validator("items")
    @classmethod
    def non_empty_items(cls, v: List[Item]) -> List[Item]:
        if not v:
            raise ValueError("items must not be empty")
        return v


# ============================
# Estado y Telemetría
# ============================

class AttemptLog(BaseModel):
    item_id: str
    attempt: int = 1
    correct: bool
    response_time_sec: float
    hints_used: int = 0
    final_score: Optional[float] = None  # 0..1
    confidence_self_report: Optional[float] = None  # 0..1
    selected_option_id: Optional[str] = None


class AdaptiveState(BaseModel):
    mastery: float = 0.6   # 0..1
    theta: float = 0.0     # -3..3
    streakCorrect: int = 0
    streakIncorrect: int = 0
    usedItems: Dict[str, int] = Field(default_factory=dict)
    lastDifficulty: Optional[DifficultyLevel] = None


# ============================
# Utilidades matemáticas
# ============================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def mastery_to_numeric(m: float) -> int:
    if m < 0.6:
        return 1
    if m < 0.85:
        return 2
    return 3


def numeric_to_level(n: int) -> DifficultyLevel:
    return "easy" if n == 1 else ("medium" if n == 2 else "hard")


def irt_p(theta: float, a: float, b: float, c: float) -> float:
    z = a * (theta - b)
    logistic = 1.0 / (1.0 + exp(-z))
    return c + (1 - c) * logistic


def update_mastery_ema(prev: float, correct: bool, rt: float, target: float = 60.0, alpha: float = 0.25) -> float:
    # Bono/penalización por tiempo (acotado)
    time_ratio = clamp(target / max(rt, 1.0), 0.75, 1.25)
    signal = (1.0 if correct else 0.0) * time_ratio
    return clamp((1 - alpha) * prev + alpha * signal, 0.0, 1.0)


def update_theta(theta: float, item: Item, correct: bool, lr: float = 0.05) -> float:
    if not item.irt_params:
        return theta
    a, b, c = item.irt_params.a, item.irt_params.b, item.irt_params.c
    p = irt_p(theta, a, b, c)
    y = 1.0 if correct else 0.0
    eps = 1e-9
    # Gradiente (aprox) de log-verosimilitud
    d = (y - p) * a * (1 - c) * p * (1 - p) / (p * (1 - p) + eps)
    return clamp(theta + lr * d, -3.0, 3.0)


# ============================
# Motor adaptativo
# ============================

def init_state(pkg: PackageLike) -> AdaptiveState:
    seed = int(clamp(float(pkg.student_state.estimated_level or 1), 1, 3))
    seed_mastery = 0.5 if seed == 1 else (0.7 if seed == 2 else 0.85)
    return AdaptiveState(
        mastery=seed_mastery,
        theta=0.0,
        streakCorrect=pkg.student_state.recent_performance.streak_correct or 0,
        streakIncorrect=pkg.student_state.recent_performance.streak_incorrect or 0,
        usedItems={},
        lastDifficulty=None,
    )


def select_next_item(pkg: PackageLike, state: AdaptiveState, attempts: List[AttemptLog]) -> Optional[Item]:
    dc = pkg.delivery_constraints
    if len(attempts) >= dc.max_items_this_session:
        return None

    # 1) Dificultad objetivo por maestría
    target_num = mastery_to_numeric(state.mastery)
    target_lvl = numeric_to_level(target_num)

    # 2) Escalación por rachas
    if state.streakCorrect >= 2 and target_num < 3:
        target_num += 1
        target_lvl = numeric_to_level(target_num)
    if state.streakIncorrect >= 2 and target_num > 1:
        target_num -= 1
        target_lvl = numeric_to_level(target_num)

    # 3) Filtrar candidatos por dificultad y reintentos
    candidates: List[Item] = []
    for it in pkg.items:
        tries = state.usedItems.get(it.item_id, 0)
        if tries >= dc.retry_policy.max_attempts_per_item:
            continue
        if it.difficulty.level == target_lvl:
            candidates.append(it)

    # 4) Fallback: si no hay candidatos, elegir no agotados cercanos a la dificultad
    if not candidates:
        open_items = [it for it in pkg.items if state.usedItems.get(it.item_id, 0) < dc.retry_policy.max_attempts_per_item]
        if not open_items:
            return None
        open_items.sort(key=lambda it: abs(int(it.difficulty.numeric) - target_num))
        candidates = open_items[:3]  # tomar algunos cercanos

    # 5) Scoring por lagunas e IRT
    policies = pkg.adaptive_policies or AdaptivePolicies()
    gaps = set(pkg.student_state.known_gaps or [])

    def score(it: Item) -> float:
        s = 1.0
        # Penalizar repetidos
        s -= 0.15 * state.usedItems.get(it.item_id, 0)
        # Sesgo por lagunas (heurística)
        if policies.use_known_gaps_biasing and gaps:
            skill_text = " ".join(it.skills).lower()
            for g in gaps:
                gl = g.lower()
                if "norma1" in gl and "norma2" in skill_text:
                    s += 0.3
                if "confusion" in gl and "definiciones" in skill_text:
                    s += 0.2
        # IRT routing: preferir p(theta) ~ 0.7
        if policies.irt_routing and it.irt_params:
            p = irt_p(state.theta, it.irt_params.a, it.irt_params.b, it.irt_params.c)
            s += 0.3 * (1 - abs(p - 0.7))
        return s

    candidates.sort(key=score, reverse=True)
    return candidates[0] if candidates else None


def apply_attempt(pkg: PackageLike, state: AdaptiveState, item: Item, attempt: AttemptLog) -> AdaptiveState:
    # Contador
    state.usedItems[item.item_id] = state.usedItems.get(item.item_id, 0) + 1

    # Rachas
    if attempt.correct:
        state.streakCorrect += 1
        state.streakIncorrect = 0
    else:
        state.streakIncorrect += 1
        state.streakCorrect = 0

    # Maestría (EMA con tiempo)
    target = item.time_limit_sec or 60
    state.mastery = update_mastery_ema(state.mastery, attempt.correct, attempt.response_time_sec, target, 0.25)

    # Theta (IRT opcional)
    policies = pkg.adaptive_policies or AdaptivePolicies()
    if policies.irt_routing and item.irt_params:
        state.theta = update_theta(state.theta, item, attempt.correct, 0.05)

    state.lastDifficulty = item.difficulty.level
    return state


def should_stop(pkg: PackageLike, state: AdaptiveState, attempts: List[AttemptLog]) -> bool:
    if len(attempts) >= pkg.delivery_constraints.max_items_this_session:
        return True
    if state.mastery >= pkg.delivery_constraints.target_mastery:
        return True
    return False


# ============================
# API (FastAPI)
# ============================

app = FastAPI(title="Adaptive Engine MVP (Python, Pydantic v2)")

# Almacén en memoria SOLO para demo
SESSIONS: Dict[str, Dict[str, object]] = {}


class InitBody(BaseModel):
    pkg: PackageLike


@app.post("/session/init")
def session_init(body: InitBody):
    pkg = body.pkg
    state = init_state(pkg)
    SESSIONS[pkg.session_id] = {"pkg": pkg, "state": state, "attempts": []}
    return {"ok": True, "state": state}


@app.get("/session/{session_id}/next")
def session_next(session_id: str):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")
    item = select_next_item(s["pkg"], s["state"], s["attempts"])  # type: ignore[index]
    if item is None:
        return {"done": True}
    return {"done": False, "item": item}


class AttemptBody(BaseModel):
    attempt: AttemptLog


@app.post("/session/{session_id}/attempt")
def session_attempt(session_id: str, body: AttemptBody):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")

    # Buscar item
    current: Optional[Item] = None
    for it in s["pkg"].items:  # type: ignore[index]
        if it.item_id == body.attempt.item_id:
            current = it
            break
    if current is None:
        raise HTTPException(400, "unknown item_id")

    s["state"] = apply_attempt(s["pkg"], s["state"], current, body.attempt)  # type: ignore[index]
    s["attempts"].append(body.attempt)  # type: ignore[index]
    return {
        "state": s["state"],
        "should_stop": should_stop(s["pkg"], s["state"], s["attempts"]),  # type: ignore[index]
    }
