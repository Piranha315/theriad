# Theriad API Documentation

## Core Interfaces

### `theriad.core.routing`

#### `RoutingDecision`
```python
@dataclass(frozen=True)
class RoutingDecision:
    tier: str           # abstract routing tier, resolved through the registry
    confidence: float   # [0.0, 1.0]
    reason: str         # human-readable explanation
```

#### `RoutingResult`
```python
@dataclass
class RoutingResult:
    decision: RoutingDecision
    layer_name: str
    all_decisions: list[RoutingDecision]   # all fired decisions (for audit)
    features: dict[str, Any]               # extracted features
```

#### `RoutingRule` (ABC)
```python
class RoutingRule(ABC):
    @property
    def name(self) -> str: ...

    def evaluate(self, prompt: str) -> RoutingDecision | None: ...
```

#### `RoutingLayer` (ABC)
```python
class RoutingLayer(ABC):
    @property
    def name(self) -> str: ...

    def route(self, prompt: str, context: dict) -> RoutingResult | None: ...
```

### `theriad.routing.v2`

`RoutingDecision` remains an abstract tier decision. Concrete Router V2 model
selection is exposed through `PolicyResolution` diagnostics and the following
internal prediction contract:

```python
@dataclass(frozen=True)
class RouterPrediction:
    model: str
    eligible: bool
    exclusion_reason: str | None
    p_success: float
    p_success_lower: float
    expected_output_tokens: float
    expected_cost_usd: float
    expected_latency_ms: float
    uncertainty: float
```

`PolicyResolution` adds `router_version`, `lane`, `knee_threshold`,
`candidate_scores`, `ood_score`, `selected_propensity`, `shadow`, and
`shadow_challenger`. These fields are empty when Router V2 is disabled or its
artifact fails validation.

---

### `theriad.core.providers`

#### `Message`
```python
@dataclass
class Message:
    role: str      # "system" | "user" | "assistant"
    content: str
```

#### `CompletionRequest`
```python
@dataclass
class CompletionRequest:
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float = 0.0
    extra: dict = {}
```

#### `CompletionResponse`
```python
@dataclass
class CompletionResponse:
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    provider: str
    raw: dict = {}
```

#### `ProviderProtocol` (ABC)
```python
class ProviderProtocol(ABC):
    @property
    def name(self) -> str: ...

    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...

    async def health_check(self) -> bool: ...
```

---

### `theriad.core.config`

#### `TheriadConfig`
```python
class TheriadConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    router: RouterConfig
    logging: LoggingConfig
    metrics_path: str
```

#### `RouterConfig`
```python
class RouterConfig(BaseModel):
    min_confidence: float = 0.75
    fallback_tier: str = "cheap_bulk"
    layers: list[str] = ["layer0", "layer1_features"]
```

---

## Layers

### `theriad.layers.layer0`

**DeterministicLayer** — runs rules in order, stops at first high-confidence hit.

Built-in rules:
- `PromptLengthRule` — short (<50w) → cheap; long (>1500w) → heavy
- `JsonDetectionRule` — JSON input → cheap
- `CodeDetectionRule` — code fences / keyword density → code model
- `MarkdownDetectionRule` — markdown structure → cheap
- `LanguageDetectionRule` — non-English → multilingual model
- `RegexRule` — custom regex matching

### `theriad.layers.layer1_features`

**FeatureExtractor** — extracts `PromptFeatures`:
- `token_count`, `word_count`, `line_count`, `char_count`
- `punctuation_ratio`, `code_fence_count`, `inline_code_count`
- `code_keyword_count`, `code_ratio`
- `markdown_heading_count`, `markdown_list_count`, `markdown_ratio`
- `json_brace_ratio`, `json_probability`

### `theriad.layers.layer2_embedding`

**EmbeddingRoutingLayer** — pluggable embedding backend, nearest-prototype routing.

Backends:
- `SentenceTransformerBackend(model_name, matryoshka_dim)` — supports BERT, MiniLM, Matryoshka
- `MockEmbeddingBackend(dim)` — deterministic mock for testing

### `theriad.layers.layer3_classifier` *(optional, removable)*

**ClassifierRoutingLayer** — TF-IDF + LogisticRegression. Only invoked if embedding confidence is below threshold. Remove safely by excluding from `RouterConfig.layers`.

---

## Engine

### `theriad.engine.pipeline.TheriadPipeline`

```python
pipeline = TheriadPipeline(
    layers=[layer0, layer1],
    providers={"openrouter": provider},
    fallback_tier="cheap_bulk",
    fallback_provider="openrouter",
    policy=ExecutionPolicy(
        timeout_seconds=30.0,
        retry=RetryPolicy(max_attempts=3, backoff_base_seconds=0.5),
        fallback=FallbackPolicy(chain=[("ollama", "llama3")]),
    ),
)

# Route only
result: RoutingResult = pipeline.route("Write a Python function")

# Route + execute
response: CompletionResponse = await pipeline.execute(messages)
```

---

## Cost Accounting

### `theriad.accounting.tracker.CostTracker`

```python
tracker = CostTracker(metrics_path="~/.theriad/metrics.jsonl")

tracker.record(
    provider="openrouter",
    model="poolside/laguna-xs-2.1:free",
    prompt_tokens=150,
    completion_tokens=80,
    latency_ms=320.0,
    routing_layer="layer0",
    routing_reason="json_detection",
)

stats = tracker.summary()
# {total_requests, total_tokens, total_cost_usd, avg_latency_ms, fallback_rate, models_used}

# Load historical records from disk
records = CostTracker.load_from_disk("~/.theriad/metrics.jsonl")
```
