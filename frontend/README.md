# Tarsiq routing preview

Set the OpenRouter key and start the local server:

```bash
export OPENROUTER_API_KEY="..."
training/.venv/bin/python frontend/server.py
```

Open http://127.0.0.1:8787. The API key remains server-side. The classifier model is read from `routing/config/routing.yaml`.
