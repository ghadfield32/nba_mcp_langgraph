# NBA Machine Learning Prediction (MCP) Platform with LangGraph Integration

A state-of-the-art FastAPI application for NBA data analysis and prediction leveraging LangGraph's AI agent workflows. This platform combines real-time NBA data with advanced machine learning techniques to deliver insights and predictions.

## 🌟 Features

- **NBA Data Integration**
  - Real-time NBA game data and statistics
  - Historical player and team performance tracking
  - Live scoreboard and play-by-play analysis
  - Comprehensive league leaders and career statistics

- **Production-Ready Architecture**
  - FastAPI for high-performance async API endpoints
  - LangGraph integration for AI agent workflows
  - Langfuse for LLM observability and monitoring
  - Structured logging with environment-specific formatting
  - Rate limiting with configurable rules
  - PostgreSQL for data persistence
  - Docker and Docker Compose support
  - Prometheus metrics and Grafana dashboards for monitoring

- **Advanced NBA Analytics**
  - Player career statistics analysis
  - Team performance metrics
  - League leaders tracking
  - Game log data exploration
  - Play-by-play breakdown and analysis

- **Security**
  - JWT-based authentication
  - Session management
  - Input sanitization
  - CORS configuration
  - Rate limiting protection

- **Developer Experience**
  - Environment-specific configuration
  - Comprehensive logging system
  - Clear project structure
  - Type hints throughout
  - Easy local development setup

- **Model Evaluation Framework**
  - Automated metric-based evaluation of model outputs
  - Integration with Langfuse for trace analysis
  - Detailed JSON reports with success/failure metrics
  - Interactive command-line interface
  - Customizable evaluation metrics

## 🏷️ Ports & Endpoints

| Service            | Port    | Purpose / Endpoint                                 |
| ------------------ | ------- | -------------------------------------------------- |
| **App (FastAPI)**  | 8000    | Main API & MCP SSE/WSS (default “claude” mode)     |
|                    | 8001    | Alternative “local” SSE/WSS mode                    |
|                    | 8000    | Swagger UI: `http://localhost:8000/docs`           |
| **PostgreSQL DB**  | 5432    | `db` service; connection string:                   |
|                    |         | `postgresql://postgres:mysecretpw@db:5432/nba_mcp_dev` |
| **Prometheus**     | 9090    | Metrics scrape target: `http://app:8000/metrics`   |
| Prometheus config: |         | `prometheus/prometheus.yml`                        |
| **Grafana**        | 3000    | Dashboards: `http://localhost:3000`<br/>Admin/admin |



## 🚀 Quick Start

### 1. Prerequisites

- Python 3.13+
- PostgreSQL
- Docker & Docker Compose (optional)

### 2. Clone & Environment Setup

```bash
git clone https://github.com/ghadfield32/nba_mcp_langgraph.git
cd nba_mcp_langgraph
```

Create and activate your virtual environment:

```bash
uv sync           # Creates and activates the .venv (cross-platform)
```

> **Note:** `uv sync` will auto-activate the venv in most shells, including PowerShell and Bash.  
> If it doesn't, you can manually activate:

- **Windows (PowerShell)**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **macOS/Linux (Bash/Zsh)**
  ```bash
  source .venv/bin/activate
  ```

Copy the example environment file and update values **for your own setup** (do **not** commit your personal secrets):

```bash
cp .env.example .env.development    # or .env.staging / .env.production
```

Open the newly created `.env.development` file and update ONLY the placeholder values:

```dotenv
APP_ENV=development
NBA_MCP_PORT=8000
POSTGRES_URL=postgresql://postgres:mysecretpw@db:5432/nba_mcp_dev
LLM_API_KEY=<your-llm-key>
JWT_SECRET_KEY=<your-jwt-secret>
```### 3. Database Setup Database Setup

Start the database:
```bash
docker compose up -d db
```
Verify it’s healthy:
```bash
docker ps  # look for "nba-db-dev" on port 5432 with healthy status
```

> If needed, manually apply `schema.sql`:
> ```bash
> sqlite3 nba_mcp_dev.db < schema.sql
> ```

### 4. Running the Application

#### Development Mode (with auto-reload)

```bash
# option 1: Invoke task
inv dev

# option 2: Makefile\make dev
```

- **Environment**: `.env.development`
- **Port**: 8000
- **Reload**: enabled (`--reload`)
- **Logging**: DEBUG, human-readable console

Watch the console for:
```
INFO: Will watch for changes in these directories...
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
Loading environment: Environment.DEVELOPMENT
Loaded environment from .env.development
...application startup complete.
```

#### Production Mode (no reload)

```bash
# locally without Docker
inv prod       # or make prod
```

- **Environment**: `.env.production`
- **Port**: 8000
- **Reload**: disabled
- **Logging**: WARNING+, JSON format

Or bring up the full stack via Docker Compose:

```bash
APP_ENV=production docker compose up -d --build
```

This will start:
- **app** (FastAPI + MCP) on port 8000
- **db** (PostgreSQL) on port 5432
- **prometheus** on port 9090
- **grafana** on port 3000


### 5. Accessing Dashboards & Endpoints

- **Swagger UI**: http://localhost:8000/docs
- **FastAPI Root**: http://localhost:8000/
- **Prometheus**: http://localhost:9090/targets (scrapes `/metrics`)
- **Grafana**: http://localhost:3000 (admin/admin)
- **Raw Metrics**: http://localhost:8000/metrics

### Grafana Dashboards

We pre‑load four dashboards under Grafana:

- **API Performance** (`api_performance.json`):  
  HTTP QPS, 95th‑pct latency, errors by endpoint.
- **Rate Limiting** (`rate_limiting.json`):  
  Per‑endpoint and global rate‑limit counters.
- **Database Performance** (`db_performance.json`):  
  Connection pool usage, active connections, query timing.
- **System Resource Usage** (`system_usage.json`):  
  Host CPU, memory, disk I/O, network (requires node‑exporter).
---
#### Automatic Provisioning

If you prefer automatic loading (recommended for production), copy the `grafana/provisioning/` directory:
```bash
cp -r ../../fastapi-langgraph-agent-production-ready-template/grafana/provisioning ./grafana/provisioning
cp -r ../../fastapi-langgraph-agent-production-ready-template/grafana/dashboards ./grafana/dashboards
```
Then update docker-compose.yml to mount:

  grafana:
    …
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards

Restart Grafana:

docker compose restart grafana

Your dashboards will now automatically appear under Dashboards → Home.
Manual Import

Alternatively, you can import JSON manually via ＋ → Import in the Grafana UI, selecting Prometheus as the data source.


---

## 💡 5. Single Versus Multiple Dashboards?

We recommend **using Grafana as a single pane of glass**—it can host all four dashboards (plus any future ones) in one UI. That keeps everything centralized:

- **One login**, **one data source configuration**, and **consistent alerts**.
- Easily create a **“folder”** inside Grafana called “NBA MCP” to group these dashboards.

---

With these README updates, new team members can immediately spin up our full monitoring stack and have rich, production‑grade dashboards at their fingertips.



## 🛠️ Infrastructure Components

### PostgreSQL
PostgreSQL is a powerful, open‑source object‑relational database management system with full ACID compliance, extensibility, and robust concurrency control, making it ideal for production environments. :contentReference[oaicite:0]{index=0}  
It offers advanced features like JSON data types, full‑text search, and multiple index methods to handle diverse data workloads efficiently. :contentReference[oaicite:1]{index=1}  
Its active community and rich ecosystem ensure continuous improvements, a wealth of extensions (e.g., PostGIS), and long‑term reliability. :contentReference[oaicite:2]{index=2}  

### Uvicorn (uv)
Uvicorn is a lightning‑fast ASGI server implementation for Python designed to run frameworks like FastAPI. :contentReference[oaicite:3]{index=3}  
It implements the ASGI specification using `uvloop` and `httptools`, supports HTTP/2 and WebSockets, and enables true asynchronous request handling for modern APIs. :contentReference[oaicite:4]{index=4}  

### Prometheus
Prometheus is an open‑source monitoring system and time‑series database originally built at SoundCloud. :contentReference[oaicite:5]{index=5}  
It scrapes metrics from instrumented applications via a pull model, storing them efficiently with a multi‑dimensional data model. :contentReference[oaicite:6]{index=6}  
Prometheus includes a powerful query language (PromQL) for real‑time aggregation, alerting, and visualization. :contentReference[oaicite:7]{index=7}  
Its standalone architecture ensures high reliability, minimal external dependencies, and straightforward setup for microservices monitoring. :contentReference[oaicite:8]{index=8}  

### Grafana
Grafana is an open‑source analytics and visualization platform that lets you create interactive, dynamic dashboards. :contentReference[oaicite:9]{index=9}  
It integrates seamlessly with multiple data sources—like Prometheus, PostgreSQL, and more—allowing you to correlate metrics across your stack. :contentReference[oaicite:10]{index=10}  
Grafana’s rich ecosystem of plugins, templating, and built‑in alerting enables tailored monitoring views and proactive operational insights. :contentReference[oaicite:11]{index=11}  


## 🚀 Standalone Ollama + LangGraph Demo

```bash
# 1. Start Ollama:
ollama serve --port 11434

# 2. In one terminal, run your NBA MCP server:
inv dev  # or `python -m nba_mcp --transport sse`

# 3. In another terminal, run the demo:
python examples/langgraph_ollama_agent_w_tools.py --mode local


## 🛠️ Adding New MCP Tools

If you’d like to extend the NBA MCP server with your own tools:

1. **Define a Pydantic model** for any structured parameters (optional):
   ```python
   from pydantic import BaseModel, Field

   class MyToolParams(BaseModel):
       team: str = Field(..., description="Team abbreviation like 'LAL'")
       limit: int = Field(10, ge=1, description="Number of records")
   ```

2. **Add a new tool function** in `nba_server.py`:
   ```python
   @mcp_server.tool()
   async def get_top_players(params: MyToolParams) -> str:
       # Your logic here, e.g. call NBAApiClient
       data = await client.get_top_players(params.team, limit=params.limit)
       return json.dumps(data)
   ```

3. **Mount it to HTTP** (optional) by adding a FastAPI route:
   ```python
   @router.get("/top_players/{team}")
   async def top_players(team: str, limit: int = Query(10)):
       params = MyToolParams(team=team, limit=limit)
       return await get_top_players(params)
   ```

4. **Reload the server** (in dev) or rebuild your Docker image (in prod).

Your new tool will now be discoverable via MCP’s `/messages/` SSE/WSS and as a standard HTTP endpoint under `/api/v1/mcp/nba/`.

