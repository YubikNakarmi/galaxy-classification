# API reference

The API layer exposes authentication, synchronous and asynchronous
classification workflows, prediction history, and WebSocket updates.

## Endpoints

- [Authentication](auth.md): sign-in and token-related behavior.
- [Classification](classify.md): submit images and create classification tasks.
- [History](history.md): retrieve previous predictions and task results.
- [WebSocket](websocket.md): receive live task and batch status updates.

The HTTP service is implemented with FastAPI. Start the local API and inspect
the generated OpenAPI documentation at `/docs` or `/redoc`.