"""Market agent modules split out from the legacy unified entrypoint."""

from pathlib import Path

from dotenv import load_dotenv


# Some submodules derive module-level defaults from environment variables during
# import. Load the project .env before those submodules initialize.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
