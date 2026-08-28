from __future__ import annotations

import os

import uvicorn

from market_agent.backend.api import create_app


if __name__ == "__main__":
    uvicorn.run(
        create_app(),
        host=str(os.getenv("MARKET_AGENT_API_HOST", "127.0.0.1")),
        port=int(os.getenv("MARKET_AGENT_API_PORT", "8080")),
        log_config=None,
    )
