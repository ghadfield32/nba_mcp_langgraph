FastMCP v2 packages three distinct HTTP servers inside one FastAPI/ASGI application; they’re just **mounted sub-apps**, so you can absolutely listen on **one port** if you want—separating ports is only a convenience for local debugging.

| Server (mount-path) | Purpose | Default started by `fastmcp dev/run` | Typical port |
|---------------------|---------|--------------------------------------|--------------|
| **MCP SSE transport** (`/mcp`) | Long-lived Server-Sent-Events stream that carries every MCP request/response between the client (Claude Desktop, your LangGraph agent, etc.) and the server. citeturn1view0 | ✅ | 8000 |
| **OpenAPI/FastAPI REST** (`/docs`, `/openapi.json`) | Auto-generated REST façade used when you convert an existing FastAPI/OpenAPI app into MCP tools/resources, or when you proxy a FastAPI backend. citeturn1view0 | ✅ | 8000 |
| **MCP Inspector UI** (`/inspector` – served by Vite dev server) | A React dashboard that lets you click through tools, resources, and live logs while you’re in *dev* mode. It’s proxied through the same ASGI app, but the CLI also spins up an independent hot-reload server on port 5173 for faster front-end rebuilds. citeturn7search3 | ✅ in `fastmcp dev` only | 5173 (configurable) |

Because SSE is just a specialised HTTP response, you can co-host it with JSON/REST endpoints under a single Uvicorn instance; frameworks such as **sse-starlette** make this a one-liner in FastAPI citeturn3search3turn3search0.  
Running two ports (e.g. 8000 REST + 8001 SSE) is useful when:

* you need different **CORS**, TLS, or buffering rules for the stream;
* you front-load with Nginx/Traefik and want independent health checks; or
* a reverse proxy (Cloudflare, AWS ALB) times out streaming paths unless routed to a dedicated upstream. citeturn3search6  

Otherwise, a single `uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2` is sufficient. Multiple FastAPI apps can be **mounted** under one root, so your LangGraph SSE router, an upload micro-service, and the Inspector UI can all live together. citeturn3search2

---

## Configuration knobs that place everything on one port

```python
# server.py
from fastmcp import FastMCP

mcp = FastMCP(
    "NBA-MCP",
    settings=dict(             # full list shown in Apidog blog
        host="0.0.0.0",
        port=8000,             # one port for every sub-app
        transport="sse",       # could be "stdio", "ws", …
        log_level="info",
    )
)

# … tools/resources …

if __name__ == "__main__":
    mcp.run()                  # single-port production run
```
`mcp.run()` is a very thin wrapper around `uvicorn.run`; you can still pass any Uvicorn option (workers, reload, SSL certs, etc.). citeturn7search6

---

## Mapping FastMCP to **dev → stage → prod**

| Environment | Primary goal | Where the LLM runs | Recommended FastMCP CLI | Port layout |
|-------------|--------------|--------------------|-------------------------|-------------|
| **Development** | Fast iteration, live reload, verbose logging | **Local** Ollama or llama.cpp so you pay \$0 while hacking | `fastmcp dev server.py` (adds Inspector, auto-reload) citeturn1view0 | 8000 (API + SSE) + 5173 (hot-reload UI) |
| **Staging** | “Dress rehearsal” that **mirrors production** infra & secrets, but is safe to break | Same cloud model as prod (Claude, OpenAI), but on non-prod API keys/quotas | `fastmcp run server.py --log-level debug` inside a Docker compose stack identical to prod | Usually one port behind a reverse proxy; may expose 8000 → 443 via TLS off-loader |
| **Production** | Real users, strict SLAs, observability, IaC | Managed LLM endpoints (Claude, OpenAI) or GPU-backed vLLM | Container or K8s deployment with health probes; run FastAPI under Gunicorn+Uvicorn workers | Fronted by TLS (443). SSE and REST share the same vhost/path, simplifying load balancer rules |

> **Local vs. prod** in FastMCP simply dictates *where the model and secrets live*—nothing in MCP forces you to change code. Your server stays identical; you swap environment variables (`MODEL_PROVIDER=openai`, `OPENAI_API_KEY=...`) and perhaps increase Uvicorn workers. This classic three-tier flow (dev → staging → prod) is an industry best-practice citeturn0search5turn4search1.

---

### Step-by-step deploy recipe

1. **Docker-ise once**. Put your FastMCP app and model adapters (OpenAI SDK, Ollama client) in one image so dev/stage/prod share the same artifact.  
2. **Use `.env` or Helm values** to switch providers:  
   * `LLM_PROVIDER=ollama   OLLAMA_BASE_URL=http://ollama:11434` in dev.  
   * `LLM_PROVIDER=openai  OPENAI_API_KEY=…` in stage/prod.  
3. **Expose one port** (`EXPOSE 8000`) and let reverse proxies (Traefik, Nginx Ingress) terminate TLS and fan-out to the container.  
4. **Add health checks**: `GET /healthz` (fast), and an SSE probe (`curl -N http://app/mcp | head -n1`) to ensure streams initialise.  
5. **CI pipeline** pushes Docker tags `:dev` `:staging` `:prod`; promotion is a tag move, not a rebuild, keeping images identical across environments. citeturn7search7turn7search9  

---

## Key take-aways

* **Yes, one port is fine**—SSE, REST, and the Inspector are all ASGI routes; split ports only if your infra demands it.  
* FastMCP’s `transport="sse"` is what actually starts the stream handler; everything else is standard FastAPI under Uvicorn. citeturn1view0  
* Treat *development*, *staging*, and *production* as deployment **targets**, not different code paths. With Docker and env-vars, the same FastMCP app can serve local Ollama or cloud LLMs with zero code changes.  
* MCP itself is transport-agnostic; once the SSE endpoint is reachable, any compliant client (Claude Desktop, LangGraph, or your own) can attach and use your NBA tools. citeturn0news70