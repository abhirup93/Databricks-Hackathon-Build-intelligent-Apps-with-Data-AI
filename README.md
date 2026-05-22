# Insurance Customer Support — Databricks Multi-Agent Chatbot

> **Databricks Hackathon · Build Intelligent Apps with Data + AI**

A production-grade, **Supervisor-router Multi-Agent System** for insurance customer support, built entirely on Databricks. A natural-language chatbot routes customer queries through specialised AI agents — each backed by live Delta tables, Vector Search, and Claude Sonnet 4.6 — and delivers polished, context-aware answers via a Streamlit frontend hosted on Databricks Apps.

---

## Architecture

![Insurance Customer Support – Multi Agent Chatbot](./insurance_support_multi_agent_architecture_8.png)

The system is organised across **6 swimlanes**:

| # | Swimlane | What lives here |
|---|---|---|
| 1 | **User & Channel** | End user, Streamlit chat UI, Databricks Apps hosting |
| 2 | **Serving & Orchestration** | MLflow PyFunc endpoint, Supervisor/Router Agent, Specialist Agents, Final Answer Composer |
| 3 | **Tooling & Retrieval** | SQL tools, RAG retrieval, clarification/escalation logic, LangGraph checkpointer |
| 4 | **Data & Knowledge Sources** | Unity Catalog Delta tables, FAQ Delta table, Databricks Vector Search index, Lakebase/Postgres, UC Volume |
| 5 | **Platform Foundation** | Unity Catalog governance, Databricks Secrets, Claude Sonnet 4.6 endpoint, GTE Large embedding endpoint, SQL Warehouse |
| 6 | **Deployment & Operations** | Source files, MLflow UC Registry, `agents.deploy()`, Databricks App, monitoring & governance |

---

## How It Works — Runtime Flow

```
User query
  → Streamlit App (Databricks Apps)
    → Model Serving Endpoint  (MLflow PyFunc · InsuranceAgentModel)
      → Supervisor / Router Agent  (Claude Sonnet 4.6)
        → Specialist Agent  (Policy / Billing / Claims / General Help / Human Escalation)
          → Tools  (SQL REST API or Vector Search)
            → Delta Tables / FAQ index
          → Final Answer Composer
      → Response returned to Streamlit
```

**Multi-turn memory** is maintained via a **LangGraph `PostgresSaver` checkpointer** backed by **Lakebase PostgreSQL** — context like `policy_number` and `customer_id` carries forward across turns without re-asking.

---

## Specialist Agents

| Agent | Trigger | Tools used |
|---|---|---|
| **Policy Agent** | Policy details, coverage, vehicle info | `get_policy_details`, `get_auto_policy_details` |
| **Billing Agent** | Premiums, invoices, payment history | `get_billing_info`, `get_payment_history` |
| **Claims Agent** | Claim status, filing, settlements | `get_claim_status` |
| **General Help Agent** | Insurance FAQs, general concepts | `retrieve_faq` → Vector Search (GTE Large) |
| **Human Escalation Agent** | Complex / sensitive cases, explicit human request | Direct LLM handoff → END |
| **Final Answer Composer** | Always last (except escalation) | Polishes & cleans specialist response → END |

The **Supervisor** routes via a JSON decision `{next_agent, task, justification}`, enforces a max-iteration guard (force-escalates at iteration ≥ 6), and uses an `ask_user` tool to request missing identifiers (policy number, customer ID, claim ID) in a single concise question.

---

## Repository Structure

```
Databricks-Hackathon-Build-intelligent-Apps-with-Data-AI/
│
├── Multi-Agent System for Insurance Customer Care/
│   ├── insurance_support_multi_agent_system.ipynb   # Main notebook — data, agents, LangGraph, deployment
│   └── requirements.txt                             # Agent + serving dependencies
│
├── insurance-support-chatbot/
│   ├── app.py                                       # Streamlit chatbot application
│   ├── app.yaml                                     # Databricks App config (port, env vars)
│   └── requirements.txt                             # App dependencies
│
├── insurance_support_multi_agent_architecture_8.png # Architecture diagram
└── README.md
```

### Notebook 1 — `insurance_support_multi_agent_system.ipynb`

Organised into **9 sections** (run top-to-bottom):

| Section | Cells | What it does |
|---|---|---|
| 1. Environment Setup | 1–4 | Auth, schema/volume creation, Lakebase connection + PostgresSaver init |
| 2. Data Layer | 5–7 | Generate synthetic data → 6 Delta tables (1K customers, 1.5K policies, 5K bills, 300 claims) |
| 3. Vector Search & FAQ | 8–14 | Download InsuranceQA-v2, write to Delta, create GTE Large Delta Sync index |
| 4. LLM Client, Tools & Prompts | 15–18 | `mlflow.deployments` client, Spark tool functions, all prompt templates |
| 5. LangGraph Agent System | 19–24 | `GraphState`, tool schemas, all agent nodes, routing, graph compile + visualise |
| 6. Testing | 25–32 | 5 scenario tests covering all agent paths + multi-turn memory demo |
| 7. Vision Pipeline | V1–V8 | Car damage OCR, driving license OCR, claim form OCR, consistency checks, VS coverage lookup |
| 8. Deployment | D1–D10 | Write `InsuranceAgentModel`, MLflow log/register, `agents.deploy()`, poll READY, smoke tests |

### Streamlit App — `insurance-support-chatbot/`

| File | Purpose |
|---|---|
| `app.py` | Multi-session Streamlit chatbot with OAuth via `WorkspaceClient`, save/load/delete conversations, example prompts, endpoint health check |
| `app.yaml` | `streamlit run app.py --server.port $DATABRICKS_APP_PORT`; injects `AGENT_ENDPOINT_NAME` as env var |
| `requirements.txt` | `streamlit>=1.35.0`, `requests>=2.31.0`, `databricks-sdk>=0.89.0` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude Sonnet 4.6 via Databricks Model Serving (`databricks-claude-sonnet-4-6`) |
| Embeddings | GTE Large via Databricks Model Serving (`databricks-gte-large-en`) |
| Agent orchestration | LangGraph `StateGraph` 0.3.5 |
| Multi-turn memory | Lakebase PostgreSQL + `PostgresSaver` (`langgraph-checkpoint-postgres`) |
| Vector Search | Databricks Vector Search — Delta Sync index, managed embeddings |
| Data platform | Unity Catalog Delta tables (6 tables), UC Volume (HF model cache) |
| Model packaging | MLflow PyFunc (code-based, `python_model=filepath`) |
| Model deployment | `databricks-agents` — `agents.deploy()` on Serverless V5 |
| Frontend | Streamlit hosted on Databricks Apps (SNAPSHOT deploy mode) |
| Secrets management | Databricks Secrets scope (`insurance_support`) |
| SQL at serving time | Databricks SQL REST API (`/api/2.0/sql/statements`) with long-lived PAT |

---

## Prerequisites

- Databricks workspace with **Serverless V5** compute
- Unity Catalog enabled with a `main` catalog (or update `CATALOG` in Cell 1)
- Access to **Databricks Model Serving** endpoints:
  - `databricks-claude-sonnet-4-6`
  - `databricks-gte-large-en`
- A **Lakebase PostgreSQL** instance provisioned in your workspace
- A **Databricks SQL Warehouse** (note the HTTP path for Cell D7)
- Permissions: `CREATE TABLE`, `CREATE SCHEMA`, `CREATE MODEL`, `USE CATALOG` on `main`

---

## Setup & Run

### Step 1 — Clone the repo

```bash
git clone https://github.com/abhirup93/Databricks-Hackathon-Build-intelligent-Apps-with-Data-AI.git
```

### Step 2 — Import Notebook 1 into Databricks

Import `Multi-Agent System for Insurance Customer Care/insurance_support_multi_agent_system.ipynb` into your Databricks workspace.

Install dependencies on your cluster (or use the `%pip install` approach in a setup cell):

```bash
pip install -r "Multi-Agent System for Insurance Customer Care/requirements.txt"
```

**`requirements.txt` (agent notebook):**
```
databricks-sdk>=0.89.0
databricks-vectorsearch
databricks-agents
mlflow
langgraph==0.3.5
langgraph-checkpoint-postgres
psycopg[binary]
psycopg2-binary
openai==1.82.0
```

### Step 3 — Run Notebook 1 (Sections 1–8)

Execute cells top-to-bottom through **Section 8 (Deployment)**. The notebook is self-contained — it creates all Delta tables, the Vector Search index, and the Lakebase checkpointer automatically.

> **Before Cell D7**: Update `SQL_WH_HTTP_PATH` with your actual SQL Warehouse HTTP path.

### Step 4 — Deploy the Streamlit App (Notebook 2)

The Streamlit app deployment is embedded in Notebook 1 (Section 8, Cell D7 deploys the agent endpoint; the app files are managed separately).

Upload the `insurance-support-chatbot/` files to your Databricks Workspace and create a Databricks App pointing to that folder, **or** run the app deployment cells in the notebook which write and deploy the app programmatically.

**`requirements.txt` (Streamlit app):**
```
streamlit>=1.35.0
requests>=2.31.0
databricks-sdk>=0.89.0
```

---

## Key Design Decisions

**Self-contained deployed model** — `InsuranceAgentModel` (the MLflow PyFunc) contains no LangGraph or Spark dependencies. At serving time it uses the Databricks SQL REST API with a long-lived PAT (stored in Secrets) for all database queries. This keeps the serving image lean and avoids Spark cold-start latency.

**Clarification bypass** — When the user's last message matches `^(POL[0-9]{6}|CUST[0-9]{5}|CLM[0-9]{6})$` after a clarification question, `predict()` routes directly to the appropriate specialist — skipping the Supervisor round-trip entirely.

**Dual execution context** — `ask_user()` calls `input()` in notebook mode and raises `AgentClarificationNeeded` in deployed mode. The same codebase runs interactively and as a stateless endpoint without modification.

**LEFT JOIN in billing** — The deployed `_get_billing_info()` uses a LEFT JOIN so `premium_amount` is always returned even when no pending billing records exist. The notebook version uses INNER JOIN (known limitation documented in the debug section).

---

## Vision Pipeline (Bonus)

Section 7 of the notebook demonstrates a standalone multimodal claim processing pipeline using Claude Sonnet 4.6 vision:

1. **Car damage analysis** — extracts make, model, damage location/severity from a photo
2. **Driving license OCR** — extracts DL number, name, DOB, expiry, validity flag
3. **Claim form OCR** — extracts all 22 fields from a motor insurance claim form
4. **Cross-document consistency checks** — fuzzy matches vehicle details, DL vs form, policy validity at incident date
5. **Vector Search coverage lookup** — queries the FAQ index with a semantic coverage query (only if all checks pass)

---

## Author

**Abhirup Pal** — Lead Data Engineer / Architect

- Medium: [medium.com/@abhirup.pal93](https://medium.com/@abhirup.pal93)
- LinkedIn: [linkedin.com/in/abhirup-pal-776066a1](https://www.linkedin.com/in/abhirup-pal-776066a1/)

---

*Built for the Databricks Hackathon: Build Intelligent Apps with Data + AI*