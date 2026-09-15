from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("WEB_TRADE_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_TRADE_PORT", "8787") or 8787)
    uvicorn.run("web_trade.backend.web_trade.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
