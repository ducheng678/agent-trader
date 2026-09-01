from __future__ import annotations

import uvicorn

from market_agent.backend.api import create_app
from market_agent.backend.governed_bootstrap import create_governed_app_from_environment
from market_agent.backend.settings import BackendSettings


if __name__ == "__main__":
    bootstrap_settings = BackendSettings.from_env().validate()
    if bootstrap_settings.environment in {"production", "prod", "staging"}:
        app = create_governed_app_from_environment(bootstrap_settings)
    else:
        app = create_app()
    settings = app.state.container.settings
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
    )
