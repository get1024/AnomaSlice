"""Atomic UTF-8 JSON persistence shared by projects, settings, and legacy annotations."""

import json
import os
import tempfile


def write_json_atomic(json_path, data, *, indent=4):
    """Replace a UTF-8 JSON document only after its contents are on disk."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=os.path.dirname(os.path.abspath(json_path)),
            suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = stream.name
            json.dump(data, stream, indent=indent, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, json_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
