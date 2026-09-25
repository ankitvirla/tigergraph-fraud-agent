# TigerGraph Fraud Investigation Agent

An agentic fraud investigation system built for the TigerGraph Hacker House Goa challenge.

The system combines **TigerGraph graph traversal**, **TigerGraph MCP**, deterministic evidence calibration, a calibrated fraud-probability model, **GraphRAG-style grounding**, and a local **Qwen 2.5** LLM reasoning layer. A React UI exposes the investigation workflow, evidence, uncertainty, graph relationships, agent trace, recommendations, and raw outputs.

---

## Overview

The system is designed around an investigator-style workflow:

```text
Case Trigger
    ↓
Investigation Agent
    ↓
TigerGraph MCP
    ↓
Customer / Transaction / Device / Card / Case Graph Context
    ↓
Evidence Calibration
    ↓
Independent Evidence Groups
    ↓
Calibrated Probability Model
    ↓
Uncertainty Assessment
    ↓
Qwen 2.5 Explanation
    ↓
Recommended Next Action
```

The LLM is used for **reasoning, evidence synthesis, and explanation**. It does not replace deterministic graph analysis or the calibrated probability model.

---

## Key Features

- TigerGraph graph-based fraud investigation
- TigerGraph MCP integration
- GSQL graph traversal and relationship analysis
- Customer, transaction, card, device, and historical-case context
- Shared-device network investigation
- Historical fraud/cleared-case evidence
- Evidence strength calibration: strong / medium / weak / not material
- Supporting / contradicting / contextual evidence separation
- Independent evidence-group calculation
- Calibrated fraud-probability model
- Explicit uncertainty assessment
- Agent-selected next action
- Local Qwen 2.5 LLM explanation through Ollama
- React investigation dashboard
- Agent Trace
- GraphRAG reasoning view
- Raw JSON investigation output
- Batch execution across 20 benchmark cases

---

# Architecture

```text
                         ┌─────────────────────┐
                         │      React UI       │
                         │ Investigation View  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │    Backend API      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │ Fraud Investigation Agent    │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
          ┌──────────────────┐          ┌──────────────────┐
          │ TigerGraph MCP   │          │ Evidence Store   │
          └────────┬─────────┘          └────────┬─────────┘
                   │                             │
                   ▼                             ▼
          ┌──────────────────┐          ┌──────────────────┐
          │ Graph Context    │          │ Evidence         │
          │                  │          │ Calibration      │
          │ Customer         │          └────────┬─────────┘
          │ Transaction      │                   │
          │ Device           │                   │
          │ Card             │                   │
          │ Fraud Case      │                   │
          └────────┬─────────┘                   │
                   └──────────────┬──────────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ Probability Model   │
                       │ HGB + Platt         │
                       └──────────┬──────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ Uncertainty Engine  │
                       └──────────┬──────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ Qwen 2.5 / Ollama   │
                       │ Explanation Layer   │
                       └──────────┬──────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ Next Action + Trace │
                       └─────────────────────┘
```

---

# TigerGraph Graph

The main graph is:

```text
FraudInvestigationGraph
```

The graph contains:

### Vertices

- `customer`
- `Transaction`
- `Device`
- `Card`
- `FraudCase`

### Edges

```text
customer ──HAS_TRANSACTION──> Transaction

customer ──USES_DEVICE──────> Device

customer ──USES_CARD────────> Card

customer ──HAS_CASE────────> FraudCase

FraudCase ──CONTAINS────────> Transaction
```

Shared-device relationships are discovered through reverse traversal:

```text
Customer A ──USES_DEVICE──> Device X <──USES_DEVICE── Customer B
```

The application does not require a dedicated `SHARED_WITH` edge.

---

# TigerGraph MCP

The agent communicates with TigerGraph through the TigerGraph MCP server.

The local MCP configuration is intentionally excluded from GitHub because it contains machine-specific configuration.

The local setup uses:

```bash
uvx tigergraph-mcp
```

The Python integration is implemented in:

```text
tigergraph_client.py
```

The client starts the MCP process and invokes TigerGraph tools programmatically.

---

# Graph Investigation

The current agent uses graph queries for:

### Customer profile

Retrieves:

- customer
- cards
- devices
- historical cases
- transactions

### Shared-device investigation

Finds other customers connected through the customer's device profiles.

### Transaction context

Retrieves:

- transaction
- related customer
- related cards
- related devices
- historical cases
- cases containing the transaction

These graph results are exposed to the agent and UI as investigation context.

---

# Evidence Calibration

Raw observations are converted into structured evidence.

Each evidence item contains concepts such as:

```text
signal
strength
direction
independence_group
explanation
```

Evidence is separated into:

### Supporting evidence

Signals that support fraud.

### Contradicting evidence

Signals that provide evidence against the fraud hypothesis.

### Contextual evidence

Useful investigation context that does not directly establish fraud.

The system also computes **independent evidence groups**.

Only supporting evidence with `strong` or `medium` strength contributes to independent evidence groups.

Correlated signals are intentionally grouped so that multiple manifestations of the same underlying behavior do not automatically count as independent evidence.

---

# Probability Model

The final probability model uses 18 features:

```text
risk_score
strong_support_count
medium_support_count
weak_support_count
medium_contradiction_count
context_evidence_count
independent_evidence_count
card_testing
cnp
cnp_new_device
account_takeover
out_of_region
new_device
proxy_network
channel_novelty
product_novelty
repeated_historical_abuse
shared_device_profile
```

Model:

```text
HistGradientBoostingClassifier
        +
Platt / sigmoid calibration
```

The calibration layer is fitted only on the validation portion of the chronological split. The final test set remains untouched for final evaluation.

The model output is treated as a **case-level historical fraud-outcome probability estimate**, not as an independently established transaction-level truth.

---

# Model Evaluation

Final held-out test metrics:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.986844 |
| PR-AUC | 0.998618 |
| Brier Score | 0.028603 |
| LogLoss | 0.092010 |
| ECE | 0.017772 |

Test fraud prevalence was approximately **90.12%**, while the historical dataset prevalence was approximately **83.83%**.

Because the historical dataset has a high fraud prevalence, model probabilities should not automatically be interpreted as production probabilities under a different deployment population without external validation.

---

# Agent Workflow

The investigation agent follows this workflow:

```text
1. Load case
       ↓
2. Retrieve graph context
       ↓
3. Load calibrated evidence
       ↓
4. Merge graph facts + evidence
       ↓
5. Build model features
       ↓
6. Calculate calibrated probability
       ↓
7. Assess uncertainty
       ↓
8. Select next action
       ↓
9. Send grounded context to Qwen
       ↓
10. Generate investigator explanation
       ↓
11. Return structured investigation result
```

The deterministic system remains authoritative for:

- evidence
- probability
- uncertainty
- next action

Qwen is used to explain those results in investigator-friendly language.

---

# Uncertainty

The agent explicitly assesses uncertainty rather than relying only on probability.

Current logic considers factors such as:

- graph retrieval failure
- absence of supporting evidence
- number of independent evidence groups
- contradictory evidence

Example outcomes:

```text
LOW
Sufficient independent supporting evidence and no major contradiction.

MEDIUM
Limited independent evidence and/or contradictions.

HIGH
Insufficient supporting evidence or missing critical graph context.
```

---

# Next Action

The agent can recommend actions such as:

```text
retrieve_missing_context
gather_additional_evidence
review_evidence_and_network_context
prepare_investigator_review
```

This creates an investigation loop rather than a simple classification result.

---

# GraphRAG Reasoning

The reasoning layer is grounded in both structured graph context and calibrated evidence.

```text
TigerGraph
    │
    ├── Customer
    ├── Transaction
    ├── Device
    ├── Card
    └── Historical Cases
             │
             ▼
       Graph Context
             │
             ├──────────────┐
             ▼              ▼
      Evidence Store     Relationships
             │              │
             └──────┬───────┘
                    ▼
              Agent Context
                    │
                    ▼
                 Qwen 2.5
                    │
                    ▼
             Grounded Explanation
```

The LLM receives only investigation context produced by the deterministic system. The prompt explicitly instructs it not to invent evidence or modify the deterministic probability.

---

# UI

The React UI provides:

- Case selection
- Benchmark case navigation
- Live agent execution
- Risk assessment
- Evidence breakdown
- Supporting evidence
- Contradicting evidence
- Contextual evidence
- Independent evidence groups
- Graph relationships
- Investigation progression
- Real-time Agent Trace
- GraphRAG reasoning view
- Qwen explanation
- Recommended next action
- Uncertainty
- Raw JSON output
- Benchmark summary

The UI can operate using the pre-generated benchmark outputs and can also call the live FastAPI agent.

---

# Screenshots

The repository includes screenshots of the graph, investigation dashboard, agent trace, and GraphRAG reasoning view:

## TigerGraph Graph

![TigerGraph fraud investigation graph](docs/screenshots/tigergraph-graph.png)

## Investigation UI

![Fraud investigation dashboard](docs/screenshots/ui-dashboard.png)

## Agent Trace

![Agent investigation trace](docs/screenshots/agent-trace.png)

## GraphRAG Reasoning

![GraphRAG reasoning view](docs/screenshots/graphrag-reasoning.png)

---

# Benchmark Cases

The system has been executed across 20 benchmark cases:

```text
HHG-001
HHG-002
HHG-003
HHG-004
HHG-005
HHG-006
HHG-007
HHG-008
HHG-009
HHG-010
HHG-011
HHG-012
HHG-013
HHG-014
HHG-015
HHG-016
HHG-017
HHG-018
HHG-019
HHG-020
```

Three useful demonstration cases are:

### HHG-005

Demonstrates:

```text
strong evidence
→ high probability
→ low uncertainty
→ investigator review
```

### HHG-009

Demonstrates:

```text
insufficient supporting evidence
→ high uncertainty
→ gather additional evidence
```

### HHG-019

Demonstrates:

```text
multiple independent evidence groups
→ comparatively low model probability
→ uncertainty / evidence review
```

These cases demonstrate that the system does not simply return the same outcome for every investigation.

---

# Project Structure

```text
tigergraph-fraud-agent/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── public/
│   │   └── data/
│   │       ├── HHG-001.json
│   │       ├── ...
│   │       └── HHG-020.json
│   │
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   │
│   ├── package.json
│   └── package-lock.json
│
├── analysis/
│   ├── calibrated_evidence.json
│   ├── probability_training_dataset.csv
│   ├── final_probability_model_summary.json
│   └── ...
│
├── benchmark_deep_analysis.py
├── benchmark_evidence_calibration.py
├── benchmark_probability_ablation.py
├── benchmark_probability_models.py
├── build_final_probability_model.py
├── build_probability_training_dataset.py
├── fraud_investigation_agent.py
├── fraud_investigation_agent_llm.py
├── fraud_probability_calibration.py
├── generate_graph_csvs.py
├── run_all_cases.py
├── tigergraph_client.py
├── validate_calibration.py
├── .env.example
└── .gitignore
```

Large raw datasets and generated TigerGraph CSV imports are intentionally excluded from the repository.

---

# Setup

## Prerequisites

- Python 3.11+
- Node.js / npm
- TigerGraph / Savanna access
- TigerGraph MCP
- `uv` / `uvx`
- Ollama
- Qwen 2.5 model

---

## Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

The benchmark runner writes generated case outputs and `benchmark_summary.json` to `submission/outputs/`. These generated files are not part of the frontend's bundled demo data.

---

# Environment Configuration

Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Configure:

```env
TG_HOST=
TG_GRAPHNAME=FraudInvestigationGraph
TG_SECRET=

OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=
```

The application reads environment variables from the process environment. Export the values before starting the backend or running the agent:

```bash
set -a
source .env
set +a
```

Never commit `.env`.

---

# TigerGraph MCP

Ensure `uvx` is installed and available:

```bash
uvx --version
```

The MCP server can be started with:

```bash
uvx tigergraph-mcp
```

The Python client manages the MCP process for the agent workflow.

---

# Ollama / Qwen

Start Ollama locally and make sure a Qwen 2.5 model is available.

For example:

```bash
ollama list
```

The agent automatically discovers a local model containing `qwen2.5`, unless `OLLAMA_MODEL` is explicitly configured.

---

# Run the Backend

From the project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/api/health
```

Run an investigation:

```bash
curl -X POST http://localhost:8000/api/investigate/HHG-019
```

---

# Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite development URL shown by the terminal.

---

# Run the Agent Directly

```bash
python fraud_investigation_agent_llm.py HHG-019
```

JSON-only output:

```bash
python fraud_investigation_agent_llm.py HHG-019 --json-only
```

---

# Run All Benchmark Cases

```bash
python run_all_cases.py
```

The batch runner executes the 20 benchmark cases and stores investigation outputs, including `benchmark_summary.json`, under `submission/outputs/`.

---

# Reproducibility / Analysis Pipeline

The analysis pipeline contains scripts for:

```text
Dataset profiling
      ↓
Benchmark deep analysis
      ↓
Evidence calibration
      ↓
Probability training dataset
      ↓
Feature audit
      ↓
Probability model comparison
      ↓
Ablation analysis
      ↓
Final probability model
      ↓
Calibration
      ↓
Validation / audit
```

Important scripts include:

```text
profile_dataset.py
benchmark_deep_analysis.py
benchmark_evidence_calibration.py
build_probability_training_dataset.py
benchmark_probability_models.py
benchmark_probability_ablation.py
build_final_probability_model.py
audit_final_probability_model.py
audit_probability_training_data.py
validate_calibration.py
```

---

# Limitations

### Historical prevalence

The historical dataset has a high fraud prevalence. Model probabilities may not transfer directly to a production population with a different base rate.

### Case-level probability

The model estimates a case-level historical fraud outcome probability. It should not be interpreted as independently established transaction-level ground truth.

### Graph profile matching

A matching device profile does not by itself prove that two customers used the same physical device.

### Risk score

The upstream `risk_score` is retained as an input feature and is not treated as an independent fraud verdict.

### LLM

The LLM is an explanation/reasoning layer. Deterministic evidence, probability, uncertainty, and graph-derived facts remain authoritative.

---

# Security

Do not commit:

```text
.env
.mcp.json
TigerGraph secrets
API keys
local credentials
virtual environments
node_modules
raw datasets
```

The repository contains `.env.example` for configuration documentation.

---

# Future Improvements

Potential extensions include:

- richer TigerGraph graph algorithms
- additional fraud patterns
- document retrieval alongside graph retrieval
- investigator feedback loops
- persistent case memory
- human-in-the-loop action execution
- production authentication
- asynchronous investigation jobs
- model monitoring and population-shift validation
- expanded benchmark coverage

---

# License

Add the appropriate project/license information before public release.
