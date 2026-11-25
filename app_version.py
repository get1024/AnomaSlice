from pathlib import Path


APP_VERSION = Path(__file__).with_name("VERSION").read_text(encoding="utf-8").strip()
