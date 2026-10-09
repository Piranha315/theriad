"""
Benchmark source registry and corpus converter.

Turns public benchmark datasets into a vintage-tagged ``EvalItem`` corpus. The
selection below is chosen for one specific job — measuring whether a *router*
escalates appropriately — which is a different job from ranking models, and
leads to different choices.

Selection criteria
------------------
1. **Mechanically gradable.** Multiple-choice or exact-match only. An
   LLM-as-judge grader would put a model in the loop that has its own biases and
   its own cost, and every grid cell would inherit both.
2. **Spread across difficulty.** A router eval needs items that cheap models
   ace *and* items they cannot touch. A corpus of uniformly hard items measures
   nothing about escalation, because the answer is always "escalate".
3. **Spread across vintage.** Old and new sources in the same corpus, so regret
   can be broken down by vintage and the contamination question becomes
   measurable rather than argued about.

On saturation and why old benchmarks still earn a place
-------------------------------------------------------
MMLU, GSM8K, and HumanEval are saturated — frontier models cluster at 88–99%, so
they no longer separate strong models from each other. That makes them poor
*model* benchmarks and it is why leaderboards have moved on.

They remain useful here for the opposite reason. A router eval needs a
well-populated easy end: items where the cheap model genuinely suffices and
escalating is pure waste. Saturated benchmarks are precisely that population.
Their saturation is the signal, not a defect — as long as difficulty is read off
the measured grid rather than assumed from the source.

Contamination
-------------
Contamination tracks a benchmark's age and popularity, so the older sources here
are the more contaminated ones. That is recorded, not corrected for: the vintage
tag lets the report show whether router regret is flat across vintages (boundary
probably transfers) or climbing on recent items (boundary is fitting
contamination in the old ones).
"""
from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from .corpus import EvalItem

_HF_ROWS = "https://datasets-server.huggingface.co/rows"
_PAGE_SIZE = 100
_TIMEOUT = 30


# ── Prompt templates ──────────────────────────────────────────────────────────
#
# No output-token cap is set anywhere in this module. Items carry
# max_output_tokens=None so the executor sends no max_tokens at all. Reasoning
# models emit a chain before the final answer, and a cap truncates them hardest
# on the difficult items — recording "wrong" where the truth is "cut off". The
# answer-extraction regexes in theriad.benchmark.types find the answer anywhere in
# the output, so long responses cost tokens but never cost correctness.

_MC_INSTRUCTION = (
    "Answer the following multiple-choice question. "
    "End your response with the letter of the correct answer on its own line, "
    "formatted exactly as: ANSWER: <letter>\n\n"
)
_EXACT_INSTRUCTION = (
    "Answer the following question with a short, precise response. "
    "End your response with the answer on its own line, "
    "formatted exactly as: ANSWER: <answer>\n\n"
)


@dataclass(frozen=True)
class BenchmarkSource:
    """One public dataset that can be converted into eval items.

    Fields:
        name:        Short identifier, used as the item id prefix and `source`.
        hf_dataset:  HuggingFace dataset path.
        config:      Dataset config name.
        split:       Split to read.
        vintage:     Publication year — the contamination/recency covariate.
        category:    Capability axis this source exercises.
        answer_type: "multiple_choice" | "exact_match".
        question_field / answer_field / choices_field: row field names.
        distractor_fields: For datasets that store each wrong option in its own
                          column instead of a single choices list (GPQA does
                          this). Options are assembled from answer_field plus
                          these, then shuffled deterministically.
        gated:       True when HF requires accepting terms + a token.
        notes:       Why this source is in the set.
    """
    name: str
    hf_dataset: str
    vintage: int
    category: str
    answer_type: str
    question_field: str
    answer_field: str
    config: str = "default"
    split: str = "test"
    choices_field: str | None = None
    distractor_fields: tuple[str, ...] = ()
    gated: bool = False
    notes: str = ""
    context_field: str | None = None
    """Passage to prepend to the question. RACE items are reading comprehension:
    the question alone ("What did Nancy try to do before she fell over?") is
    unanswerable without the article, and a corpus built without it would grade
    every model at chance while looking perfectly well-formed."""
    choices_subfield: str | None = None
    """Key holding the option strings when ``choices_field`` is a mapping rather
    than a list. ARC stores ``{"text": [...], "label": [...]}``."""
    labels_subfield: str | None = None
    """Key holding the option labels, used to resolve the answer key. ARC labels
    are "A"-"D" on most items and "1"-"4" on others, so the answer cannot be
    assumed to be a letter — it is looked up positionally instead."""

    @property
    def assembles_choices(self) -> bool:
        """True when options must be built from separate distractor columns."""
        return bool(self.distractor_fields)


# The recommended default set: two saturated/older sources to populate the easy
# end, two modern differentiators for the hard end, spanning knowledge, reasoning
# and math. Field names must be verified against each dataset's viewer before a
# populate run — schemas drift, and a wrong field name yields a corpus of
# well-formed nonsense that grades as universal failure.
DEFAULT_SOURCES: list[BenchmarkSource] = [
    BenchmarkSource(
        name="mmlu",
        hf_dataset="cais/mmlu",
        config="all",
        split="test",
        vintage=2020,
        category="qa",
        answer_type="multiple_choice",
        question_field="question",
        answer_field="answer",
        choices_field="choices",
        notes="Saturated by 2026 — included deliberately to populate the easy end "
              "where escalation is waste. Also the most contaminated source here.",
    ),
    BenchmarkSource(
        name="gsm8k",
        hf_dataset="openai/gsm8k",
        config="main",
        split="test",
        vintage=2021,
        category="reasoning",
        answer_type="exact_match",
        question_field="question",
        answer_field="answer",
        notes="Saturated grade-school math; easy-end anchor for the reasoning axis.",
    ),
    BenchmarkSource(
        name="gpqa",
        hf_dataset="Idavidrein/gpqa",
        config="gpqa_diamond",
        split="train",
        vintage=2023,
        category="reasoning",
        answer_type="multiple_choice",
        question_field="Question",
        answer_field="Correct Answer",
        distractor_fields=(
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
        ),
        gated=True,
        notes="Graduate-level, still a live differentiator. Gated on HF. Stores "
              "each wrong option in its own column rather than a choices list, "
              "so options are assembled and shuffled — see distractor_fields.",
    ),
    BenchmarkSource(
        name="mmlu_pro",
        hf_dataset="TIGER-Lab/MMLU-Pro",
        config="default",
        split="test",
        vintage=2024,
        category="qa",
        answer_type="multiple_choice",
        question_field="question",
        answer_field="answer",
        choices_field="options",
        notes="Harder MMLU successor drawn from a wider pool; less contaminated "
              "and not yet saturated.",
    ),
]

# ── Difficulty-labelled sources ───────────────────────────────────────────────
#
# ARC and RACE each ship a *human-assigned* difficulty split — ARC-Easy vs
# ARC-Challenge, RACE middle-school vs high-school. That is rare and valuable:
# every other source here yields difficulty only as a by-product of measuring
# models against it, which is circular when the thing being trained is a
# difficulty predictor. These give an external label to validate against.
#
# They also widen the corpus beyond the four-source setup, where "difficulty"
# and "which benchmark" are perfectly confounded. RACE in particular is reading
# comprehension over a passage — a different task family from the science/maths
# multiple-choice that dominates the current corpus, which is what a predictor
# needs if it is to learn difficulty rather than topic.
#
# Caveat: both are old (2018 and 2017) and heavily contaminated. Expect the easy
# splits to be saturated for every model in the registry, contributing little
# signal at the top. They earn their place as *training data for the predictor*
# and as easy-end anchors, not as differentiators.
DIFFICULTY_LABELLED_SOURCES: list[BenchmarkSource] = [
    BenchmarkSource(
        name="arc_easy",
        hf_dataset="allenai/ai2_arc",
        config="ARC-Easy",
        split="test",
        vintage=2018,
        category="qa",
        answer_type="multiple_choice",
        question_field="question",
        answer_field="answerKey",
        choices_field="choices",
        choices_subfield="text",
        labels_subfield="label",
        notes="Grade-school science, human-labelled EASY. Saturated; anchors the "
              "easy end and pairs with arc_challenge for a controlled "
              "difficulty contrast on matched topic and format.",
    ),
    BenchmarkSource(
        name="arc_challenge",
        hf_dataset="allenai/ai2_arc",
        config="ARC-Challenge",
        split="test",
        vintage=2018,
        category="qa",
        answer_type="multiple_choice",
        question_field="question",
        answer_field="answerKey",
        choices_field="choices",
        choices_subfield="text",
        labels_subfield="label",
        notes="Same source and format as arc_easy but human-labelled HARD — "
              "retrieval and co-occurrence solvers fail these. The matched pair "
              "isolates difficulty from topic, which no other source here does.",
    ),
    BenchmarkSource(
        name="race_middle",
        hf_dataset="ehovy/race",
        config="middle",
        split="test",
        vintage=2017,
        category="qa",
        answer_type="multiple_choice",
        question_field="question",
        answer_field="answer",
        choices_field="options",
        context_field="article",
        notes="Middle-school reading comprehension. Adds a passage-grounded task "
              "family absent from the current corpus.",
    ),
    BenchmarkSource(
        name="race_high",
        hf_dataset="ehovy/race",
        config="high",
        split="test",
        vintage=2017,
        category="qa",
        answer_type="multiple_choice",
        question_field="question",
        answer_field="answer",
        choices_field="options",
        context_field="article",
        notes="High-school reading comprehension; the harder half of the RACE "
              "pair. Long prompts — these are the items where the provider-side "
              "truncation seen in the Aug 2026 grid is most likely to bite.",
    ),
]


HLE_SOURCE = BenchmarkSource(
    name="hle",
    hf_dataset="cais/hle",
    vintage=2025,
    category="qa",
    answer_type="exact_match",
    question_field="question",
    answer_field="answer",
    gated=True,
    notes="Hard end of the corpus. Use theriad.benchmark.hle_converter, which "
          "already handles HLE's answer-type split and sets no token cap.",
)


# ── Fetching ──────────────────────────────────────────────────────────────────


def _auth_headers(token: str | None) -> dict[str, str]:
    token = token or os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_rows(
    source: BenchmarkSource, limit: int, hf_token: str | None = None
) -> list[dict]:
    """Fetch up to ``limit`` rows from the HuggingFace datasets-server."""
    rows: list[dict] = []
    offset = 0
    while len(rows) < limit:
        page = min(_PAGE_SIZE, limit - len(rows))
        query = urllib.parse.urlencode(
            {
                "dataset": source.hf_dataset,
                "config": source.config,
                "split": source.split,
                "offset": offset,
                "length": page,
            }
        )
        req = urllib.request.Request(
            f"{_HF_ROWS}?{query}", headers=_auth_headers(hf_token)
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError(
                    f"{source.hf_dataset} requires authentication "
                    f"({'gated dataset' if source.gated else 'HTTP ' + str(exc.code)}).\n"
                    f"  1. Accept terms at https://huggingface.co/datasets/{source.hf_dataset}\n"
                    f"  2. Create a read token at https://huggingface.co/settings/tokens\n"
                    f"  3. Set HF_TOKEN or pass --hf-token"
                ) from exc
            raise RuntimeError(
                f"HuggingFace returned HTTP {exc.code} for {source.hf_dataset}"
            ) from exc
        batch = [entry["row"] for entry in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows[:limit]


# ── Conversion ────────────────────────────────────────────────────────────────


def _letter(index: int) -> str:
    return chr(ord("A") + index)


def row_to_item(source: BenchmarkSource, row: dict, index: int) -> EvalItem | None:
    """Convert one dataset row to an EvalItem, or None when unusable."""
    question = row.get(source.question_field)
    answer = row.get(source.answer_field)
    if not question or answer is None:
        return None

    prompt_body = str(question)
    answer_text: str

    if source.answer_type == "multiple_choice":
        if source.assembles_choices:
            # Options live in separate columns (GPQA). Build the list, then
            # shuffle so the correct answer is not always in position A —
            # models have well-documented position biases, and leaving the key
            # first would measure that bias instead of capability.
            #
            # The shuffle is seeded on the item id so a corpus is reproducible:
            # the same id always yields the same option order, which matters
            # because the grid is keyed on item id and accumulates across runs.
            distractors = [
                str(row[f]) for f in source.distractor_fields if row.get(f)
            ]
            if not distractors:
                return None
            options = [str(answer), *distractors]
            random.Random(f"{source.name}:{index}").shuffle(options)
            choices = options
            answer = str(answer)
        else:
            choices = row.get(source.choices_field) if source.choices_field else None
            labels = None
            if isinstance(choices, dict):
                # Mapping-shaped choices (ARC): pull the option text out, and
                # keep the parallel label list so the answer key can be resolved
                # positionally rather than assumed to be a letter.
                labels = choices.get(source.labels_subfield) if source.labels_subfield else None
                choices = choices.get(source.choices_subfield or "text")
            if labels and answer is not None:
                try:
                    answer = int(list(labels).index(str(answer)))
                except ValueError:
                    return None
        if not choices:
            return None
        rendered = "\n".join(
            f"{_letter(i)}. {c}" for i, c in enumerate(choices)
        )
        context = row.get(source.context_field) if source.context_field else None
        stem = f"{context}\n\n{question}" if context else str(question)
        prompt_body = f"{stem}\n\n{rendered}"
        # answer may be an index or the literal correct option
        if isinstance(answer, int):
            if not 0 <= answer < len(choices):
                return None
            answer_text = _letter(answer)
        else:
            try:
                answer_text = _letter(list(choices).index(answer))
            except ValueError:
                # Some sets store the letter directly.
                text = str(answer).strip().upper()
                if len(text) == 1 and text.isalpha():
                    answer_text = text
                else:
                    return None
        instruction = _MC_INSTRUCTION
    else:
        # GSM8K-style rationales put the final answer after '####'.
        answer_text = str(answer).split("####")[-1].strip()
        if not answer_text:
            return None
        instruction = _EXACT_INSTRUCTION

    return EvalItem(
        id=f"{source.name}_{index:05d}",
        category=source.category,
        prompt=instruction + prompt_body,
        answer=answer_text,
        answer_type=source.answer_type,
        source=source.name,
        vintage=source.vintage,
        max_output_tokens=None,  # never cap; see module docstring
    )


def build_corpus(
    sources: list[BenchmarkSource],
    per_source: int,
    hf_token: str | None = None,
    seed: int = 42,
    fetcher=fetch_rows,
) -> list[EvalItem]:
    """Fetch and convert ``per_source`` items from each source.

    ``fetcher`` is injected so this is testable without network access.
    Over-fetches to absorb rows that fail conversion, then samples down.
    """
    rng = random.Random(seed)
    corpus: list[EvalItem] = []
    for source in sources:
        rows = fetcher(source, per_source * 2, hf_token)
        items = [
            item
            for i, row in enumerate(rows)
            if (item := row_to_item(source, row, i)) is not None
        ]
        if len(items) > per_source:
            items = rng.sample(items, per_source)
        corpus.extend(items)
    return corpus


# ── Source verification ───────────────────────────────────────────────────────


@dataclass
class SourceCheck:
    """Result of probing one source's schema."""
    source: BenchmarkSource
    ok: bool
    reason: str = ""
    available_fields: list[str] = field(default_factory=list)
    sample_answer: str | None = None

    @property
    def line(self) -> str:
        mark = "OK  " if self.ok else "FAIL"
        return f"{mark} {self.source.name:<12} {self.reason}"


def verify_source(
    source: BenchmarkSource, hf_token: str | None = None, fetcher=fetch_rows
) -> SourceCheck:
    """Probe one source and confirm its declared field mapping actually works.

    Schemas drift and dataset configs get renamed. A wrong field name does not
    error — it silently yields a corpus of well-formed nonsense that grades as
    universal failure, and you only find out after paying for the grid. This
    check costs one HTTP request per source and no provider spend.
    """
    try:
        rows = fetcher(source, 1, hf_token)
    except Exception as exc:
        return SourceCheck(source=source, ok=False, reason=str(exc))

    if not rows:
        return SourceCheck(source=source, ok=False, reason="no rows returned")

    row = rows[0]
    available = sorted(row)

    declared = [source.question_field, source.answer_field]
    if source.choices_field:
        declared.append(source.choices_field)
    declared.extend(source.distractor_fields)
    missing = [f for f in declared if f not in row]
    if missing:
        return SourceCheck(
            source=source,
            ok=False,
            reason=f"missing field(s) {missing}; available: {available}",
            available_fields=available,
        )

    item = row_to_item(source, row, 0)
    if item is None:
        return SourceCheck(
            source=source,
            ok=False,
            reason=f"fields present but conversion failed; available: {available}",
            available_fields=available,
        )

    # Gradability round-trip. A converted item is not enough: the grader must be
    # able to *see* the answer in a compliant response. This caught a real bug —
    # the extractor only recognised letters A–E while MMLU-Pro keys run A–J, so
    # every F–J item graded as wrong for every model, and the grid would have
    # reported universal failure on half that source with nothing in the logs to
    # say why. Conversion succeeding tells you the corpus is well-formed; only
    # this tells you it is scoreable.
    compliant = (
        f"ANSWER: {item.answer}"
        if source.answer_type == "multiple_choice"
        else f"ANSWER: {item.answer}"
    )
    if not item.grade(compliant):
        return SourceCheck(
            source=source,
            ok=False,
            reason=(
                f"converted but UNGRADABLE: key {item.answer!r} is not recovered "
                f"from a compliant response. The grader cannot score this source."
            ),
            available_fields=available,
            sample_answer=item.answer,
        )

    return SourceCheck(
        source=source,
        ok=True,
        reason=f"answer={item.answer!r} vintage={item.vintage} gradable=yes",
        available_fields=available,
        sample_answer=item.answer,
    )


def verify_sources(
    sources: list[BenchmarkSource], hf_token: str | None = None, fetcher=fetch_rows
) -> list[SourceCheck]:
    """Verify every source. Returns checks in the order given."""
    return [verify_source(s, hf_token, fetcher) for s in sources]
