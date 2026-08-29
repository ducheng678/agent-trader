from __future__ import annotations

import uvicorn

from market_agent.backend.api import create_app


if __name__ == "__main__":
    app = create_app()
    settings = app.state.container.settings
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
    )
