# Exa Integration Plan
This document outlines the architecture for integrating Exa as the shared evidence retrieval layer for your multi-agent oracle system. The design ensures that all LLM agents (GPT-4o, Claude Haiku, Gemini 2.0) receive identical evidence, isolating reasoning capability from retrieval capability.


## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RETRIEVAL LAYER                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌─────────────────────────────────────────────────┐
│                  │      │                                                 │
│   KalshiBench    │      │              ExaOracleRetriever                 │
│                  │      │                                                 │
│  ┌────────────┐  │      │  ┌─────────────────────────────────────────┐   │
│  │ question   │──┼──────┼─▶│  question_to_query()                    │   │
│  │ criteria   │  │      │  │                                         │   │
│  │ res_date   │  │      │  │  "What is the result of this question   │   │
│  └────────────┘  │      │  │   '{question}'"                         │   │
│                  │      │  └──────────────────┬──────────────────────┘   │
└──────────────────┘      │                     │                          │
                          │                     ▼                          │
                          │  ┌─────────────────────────────────────────┐   │
                          │  │         exa.search_and_contents()       │   │
                          │  │                                         │   │
                          │  │  • type="auto"                          │   │
                          │  │  • num_results=10                       │   │
                          │  │  • end_published_date=resolution_date   │   │
                          │  │  • text={"max_characters": 4000}        │   │
                          │  └──────────────────┬──────────────────────┘   │
                          │                     │                          │
                          └─────────────────────┼──────────────────────────┘
                                                │
                                                ▼
                          ┌─────────────────────────────────────────────────┐
                          │                 EvidencePacket                  │
                          │                                                 │
                          │  • question_id                                  │
                          │  • question_text                                │
                          │  • resolution_criteria                          │
                          │  • resolution_date                              │
                          │  • retrieval_timestamp                          │
                          │  • query_used                                   │
                          │  • sources: List[Source]  ◄── Ranked by Exa     │
                          │      └─▶ Source 1 (most relevant)               │
                          │      └─▶ Source 2                               │
                          │      └─▶ ...                                    │
                          │      └─▶ Source 10 (least relevant)             │
                          └──────────────────┬──────────────────────────────┘
                                             │
                      ┌──────────────────────┼──────────────────────┐
                      │                      │                      │
                      ▼                      ▼                      ▼
              ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
              │   GPT-4o     │      │ Claude Haiku │      │  Gemini 2.0  │
              │    Agent     │      │    Agent     │      │    Agent     │
              └──────────────┘      └──────────────┘      └──────────────┘
                      │                      │                      │
                      │         IDENTICAL EVIDENCE                  │
                      │              PACKET                         │
                      └──────────────────────┴──────────────────────┘
```

---

## 1. Query Format

```python
def question_to_query(question: str) -> str:
    return f'What is the result of this question "{question.strip().rstrip("?")}"'
```

**Example:**
| Input | Output |
|-------|--------|
| `"Will the Lakers win the 2024 NBA Finals?"` | `What is the result of this question "Will the Lakers win the 2024 NBA Finals"` |

---

## 2. API Call Configuration

```python
from exa_py import Exa

exa = Exa(api_key=os.getenv('EXA_API_KEY'))

results = exa.search_and_contents(
    query=question_to_query(question_text),
    type="auto",
    num_results=10,
    end_published_date=resolution_date,  # ISO format: "2024-06-20T00:00:00.000Z"
    text={"max_characters": 4000}
)
```

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `type` | `"auto"` | Let Exa choose neural vs keyword |
| `num_results` | `10` | Good coverage without noise |
| `end_published_date` | Resolution timestamp | **Critical:** Prevents future data leakage |
| `text.max_characters` | `4000` | ~1000 words per source |

### Don't Use

| Parameter | Why Not |
|-----------|---------|
| `highlights` | Adds Exa's relevance judgment as confounding variable |
| `summary` | Abstractive content from another LLM (Gemini Flash) |
| `context` | Loses individual source metadata needed for analysis |

---

## 3. Output Structure

```python
@dataclass
class Source:
    title: str
    url: str
    published_date: Optional[str]
    text: str

@dataclass
class EvidencePacket:
    question_id: str
    question_text: str
    resolution_criteria: str
    resolution_date: str
    retrieval_timestamp: str
    query_used: str
    sources: List[Source]  # Ordered by relevance (index 0 = most relevant)
```

---

## 4. Files to Create

```
retrieval/
├── exa_retriever.py    # ExaOracleRetriever class
└── evidence.py         # Source, EvidencePacket dataclasses
```

---

## 5. Caching

Save each evidence packet as JSON for reproducibility:

```
cache/evidence/{question_id}.json
```

---

## 6. Cost Estimate

| Component | Per Question | 1500 Questions |
|-----------|--------------|----------------|
| Exa search | $0.005 | $7.50 |
| Exa content (10 pages) | $0.01 | $15.00 |
| **Total** | **$0.015** | **$22.50** |