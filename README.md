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

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+  
- PostgreSQL  
- Docker & Docker Compose  

### Environment Setup

```bash
git clone <repository-url>
cd <project-directory>
uv sync                              # create & activate venv
cp .env.example .env.development     # or .env.staging / .env.production
# Edit `.env.development`:
#   • APP_ENV=development
#   • NBA_MCP_PORT=8000               # SSE/WSS port for “claude” mode
#   • POSTGRES_URL=postgresql://postgres:mysecretpw@db:5432/nba_mcp_dev
#   • LLM_API_KEY=sk-…
#   • JWT_SECRET_KEY=…

### Database setup

    docker compose up -d db

    Verify healthy: docker ps shows nba-db-dev on 5432, healthy status

    (Optional) Manually apply schemas.sql if migrations fail

Running the Application
Local (no Docker)

uv sync
# "claude" mode → SSE/WSS on port from NBA_MCP_PORT (default 8000)
python app\services\mcp\nba_mcp\nba_server.py --mode claude --transport sse

Docker Compose

# build & start all services (db, app, prometheus, grafana)
docker compose up -d --build

# or target only the app + db:
docker compose up -d db app

Once up:

    API & MCP → http://localhost:8000

    Swagger UI → http://localhost:8000/docs

    Prometheus → http://localhost:9090

    Grafana → http://localhost:3000 (admin/admin)

## 🏀 NBA Data Endpoints

The platform provides rich endpoints for NBA data:

- `/scoreboard` - Live or historical NBA scoreboard data
- `/player/{player_name}/career_stats` - Comprehensive player career information
- `/leaders/{category}` - League leaders by statistical category
- `/gamelog` - Season game logs by season/date/team
- `/playbyplay` - Detailed play-by-play data for games


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

### Prerequisites

- Python 3.13+
- PostgreSQL ([see Database setup](#database-setup))
- Docker and Docker Compose (optional)

### Environment Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd <project-directory>
```

2. Create and activate a virtual environment:

```bash
uv sync
```

3. Copy the example environment file:

```bash
cp .env.example .env.[development|staging|production] # e.g. .env.development
```

4. Update the `.env` file with your configuration (see `.env.example` for reference)

### Database setup

1. Create a PostgreSQL database (e.g Supabase or local PostgreSQL)
2. Update the database connection string in your `.env` file:

```bash
POSTGRES_URL="postgresql://:your-db-password@POSTGRES_HOST:POSTGRES_PORT/POSTGRES_DB"
```

- You don't have to create the tables manually, the ORM will handle that for you.But if you faced any issues,please run the `schemas.sql` file to create the tables manually.

### Running the Application

#### Local Development

1. Install dependencies:

```bash
uv sync
```

2. Run the application:

```bash
make [dev|staging|production] # e.g. make dev
```

1. Go to Swagger UI:

```bash
http://localhost:8000/docs
```



#### Using Docker

1. Build and run with Docker Compose:

```bash
make docker-build-env ENV=[development|staging|production] # e.g. make docker-build-env ENV=development
make docker-run-env ENV=[development|staging|production] # e.g. make docker-run-env ENV=development
```

2. Access the monitoring stack:

```bash
# Prometheus metrics
http://localhost:9090

# Grafana dashboards
http://localhost:3000
Default credentials:
- Username: admin
- Password: admin
```

The Docker setup includes:

- FastAPI application
- PostgreSQL database
- Prometheus for metrics collection
- Grafana for metrics visualization
- Pre-configured dashboards for:
  - API performance metrics
  - Rate limiting statistics
  - Database performance
  - System resource usage

## 📊 Model Evaluation

The project includes a robust evaluation framework for measuring and tracking model performance over time. The evaluator automatically fetches traces from Langfuse, applies evaluation metrics, and generates detailed reports.

### Running Evaluations

You can run evaluations with different options using the provided Makefile commands:

```bash
# Interactive mode with step-by-step prompts
make eval [ENV=development|staging|production]

# Quick mode with default settings (no prompts)
make eval-quick [ENV=development|staging|production]

# Evaluation without report generation
make eval-no-report [ENV=development|staging|production]
```

## 🔧 NBA MCP Server Configuration

- **Interactive CLI**: User-friendly interface with colored output and progress bars
- **Flexible Configuration**: Set default values or customize at runtime
- **Detailed Reports**: JSON reports with comprehensive metrics including:
  - Overall success rate
  - Metric-specific performance
  - Duration and timing information
  - Trace-level success/failure details

### Customizing Metrics

Evaluation metrics are defined in `evals/metrics/prompts/` as markdown files:

1. Create a new markdown file (e.g., `my_metric.md`) in the prompts directory
2. Define the evaluation criteria and scoring logic
3. The evaluator will automatically discover and apply your new metric

### Viewing Reports

Reports are automatically generated in the `evals/reports/` directory with timestamps in the filename:

```
evals/reports/evaluation_report_YYYYMMDD_HHMMSS.json
```

Each report includes:

- High-level statistics (total trace count, success rate, etc.)
- Per-metric performance metrics
- Detailed trace-level information for debugging

## 🔧 Configuration

    .env.development, .env.staging, .env.production for each environment

    Key variables:

        APP_ENV — which environment profile to load

        NBA_MCP_PORT — SSE/WSS port (8000 for “claude”, 8001 for “local”)

        POSTGRES_URL — should point at db:5432 when running in Compose

        LLM_API_KEY, JWT_SECRET_KEY, etc.

## ⚙️ Docker Compose Services

services:
  db:
    image: postgres:15
    ports: ["5432:5432"]
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: mysecretpw
      POSTGRES_DB: nba_mcp_dev

  app:
    build: .
    ports: ["8000:8000"]
    depends_on:
      db:
        condition: service_healthy
    env_file:
      - .env.${APP_ENV:-development}
    environment:
      NBA_MCP_PORT: ${NBA_MCP_PORT:-8000}
      POSTGRES_URL: postgresql://postgres:mysecretpw@db:5432/nba_mcp_dev
      LLM_API_KEY: ${LLM_API_KEY}
      JWT_SECRET_KEY: ${JWT_SECRET_KEY}
    healthcheck:
      test: ["CMD","curl","-f","http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: false
