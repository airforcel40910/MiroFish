# Calibre setup: MiroFish on macOS (Apple silicon) with Claude

This fork runs MiroFish locally in Docker on an M1/M2/M3 Mac, using Anthropic's
Claude API through its OpenAI-compatible endpoint (`https://api.anthropic.com/v1/`).
It uses two models: a main model for the analysis steps and a cheaper, faster
"boost" model for the many per-agent actions during a simulation.

## Run it

Requirements: Docker Desktop for Mac (Apple silicon build), an Anthropic API key,
and a Zep Cloud API key.

```bash
git clone https://github.com/AirforceL40910/MiroFish.git
cd MiroFish
cp .env.example .env
# Edit .env: comment out the original block at the top, then uncomment the
# Anthropic block at the bottom and fill in your keys (template below).
docker compose up -d --build
```

Open http://localhost:3000 once the build finishes. The first build takes several
minutes because it installs the Python and Node dependencies.

The app is published on `127.0.0.1` only, so other devices on your network can't
reach it. Inside the container the servers still listen on `0.0.0.0`.

## `.env` template

```dotenv
LLM_API_KEY=sk-ant-your_key_here
LLM_BASE_URL=https://api.anthropic.com/v1/
LLM_MODEL_NAME=claude-sonnet-5-5

# Optional boost model for the per-agent simulation actions.
# Leave any of these blank (or delete them) to reuse the main value.
LLM_BOOST_API_KEY=
LLM_BOOST_BASE_URL=
LLM_BOOST_MODEL_NAME=claude-haiku-5-5

ZEP_API_KEY=your_zep_api_key_here

# Maximum concurrent LLM requests per process (default 4)
LLM_MAX_CONCURRENCY=4
```

Don't leave the placeholder values from the original block in place
(`LLM_BOOST_BASE_URL=your_base_url_here` and so on). A non-blank boost value
counts as configured. Never commit `.env`. It is already listed in `.gitignore`.

## Which model does what

| Step | Model |
| --- | --- |
| Ontology generation from your seed documents | main (`LLM_MODEL_NAME`) |
| Knowledge-graph extraction | done by Zep Cloud from the ontology, so neither model is used |
| Persona (agent profile) generation | main |
| Simulation configuration | main |
| Per-agent actions in each simulation round (Twitter and Reddit) | boost (`LLM_BOOST_MODEL_NAME`), or main if no boost value is set |
| Agent interviews during or after a simulation | boost, because they use the simulation's agents |
| Report generation and the ReportAgent chat | main |

### Where each setting is read

| `.env` variable | Read by |
| --- | --- |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME` | `backend/app/config.py` for every Flask-side call, through `LLMClient`, `OasisProfileGenerator` and `SimulationConfigGenerator` |
| `LLM_BOOST_API_KEY`, `LLM_BOOST_BASE_URL`, `LLM_BOOST_MODEL_NAME` | `backend/scripts/camel_llm.py` in the simulation subprocess. A blank field falls back to the matching `LLM_*` field. |
| `OPENAI_API_KEY`, `OPENAI_API_BASE_URL` | Not set by you. `camel_llm.py` fills them from the resolved LLM settings for camel-ai and camel-oasis, and also passes the key and URL to the model explicitly. |
| `LLM_MAX_CONCURRENCY` | `backend/app/utils/openai_chat_compat.py` for Flask-side calls, and `camel_llm.py` plus the OASIS env semaphore in the simulation subprocess |
| `ZEP_API_KEY` | `backend/app/config.py` |

When the base URL is Anthropic's, the backend leaves out `temperature` and
`response_format`: Claude 4.7 and later models reject any temperature other
than 1, and the endpoint ignores `response_format`. JSON replies are parsed after
stripping any Markdown code fences, and a reply that fails to parse is requested
once more. Requests to other providers are sent unchanged. Any request that gets
HTTP 429 (rate limit) or 529 (overloaded) is retried up to 6 times with
exponential backoff and jitter.

## Change models

Edit `LLM_MODEL_NAME` or `LLM_BOOST_MODEL_NAME` in `.env`, then restart (next
section). You can use any Claude model ID that your key can access, for example
`claude-opus-5-5`, `claude-sonnet-5-5` or `claude-haiku-5-5`. To run everything
on one model, leave the `LLM_BOOST_*` lines blank.

## Restart after editing `.env`

```bash
docker compose up -d --force-recreate
```

Docker reads `.env` only when it creates the container, so a plain
`docker compose restart` won't pick up your changes. After a code change, add
`--build`.

## Stop

```bash
docker compose down
```

Uploaded files and simulation data stay in `backend/uploads/` on your Mac.

## View logs

```bash
docker compose logs -f            # follow all output
docker compose logs --tail 200    # last 200 lines
```

## Troubleshooting

- **Port already in use (3000 or 5001):** another app is using the port. Stop it,
  or change the host side of the mapping in `docker-compose.yml`, for example
  `127.0.0.1:3001:3000`. The backend has to stay on 5001 because the frontend
  calls `http://localhost:5001`.
- **`LLM_API_KEY 未配置` or 401 errors:** check that `.env` is next to
  `docker-compose.yml`, that the Anthropic block is uncommented, and that the key
  has no quotes or spaces around it. Then run `docker compose up -d --force-recreate`.
- **404 or "model not found":** the model ID is wrong or your key can't access
  that model. Check that `LLM_BASE_URL` ends with `/v1/`.
- **Frequent 429 or 529 errors in the logs:** you are hitting your account's rate
  limit, or Anthropic is overloaded. Lower `LLM_MAX_CONCURRENCY` to 2, reduce the
  number of agents or rounds, or use a lower-volume boost model.
- **The simulation uses the wrong model:** look for `[加速LLM]` (boost) or
  `[通用LLM]` (main) in `docker compose logs`. Each one is followed by the model
  name in use.
- **Build fails while downloading dependencies:** retry `docker compose build`.
  Docker Hub and GitHub Container Registry downloads sometimes fail temporarily.
- **The page loads but API calls fail:** run `docker compose ps` to confirm the
  container is up, and check `docker compose logs` for a Python traceback.
- **Starting fresh:** run `docker compose down`, then `docker compose up -d --build`.
  To also delete past projects, empty `backend/uploads/` but keep `.gitkeep`.
