"""Generate the README config table from the plugin config class.

The table used to be hand-written and had drifted: a third of the fields were
missing and two defaults were stale. Deriving it means the docs cannot lie
about a default again, and `tests/test_docs.py` fails when the two diverge.

    python tools/gen_config_table.py          # print the table
    python tools/gen_config_table.py --write  # rewrite README.md in place
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.hoststub import install_host_stub  # noqa: E402

install_host_stub()

import nekro_auto_sleep as plugin_module  # noqa: E402

BEGIN = "<!-- config-table:begin -->"
END = "<!-- config-table:end -->"


def _describe(field) -> str:
    extra = field.json_schema_extra or {}
    i18n = extra.get("i18n_description") or {}
    text = i18n.get("zh-CN") or field.description or ""
    return " ".join(str(text).split())


def _render_default(value) -> str:
    if value is None or value == "":
        return "空"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    return f"`{value}`"


def build_table() -> str:
    rows = ["| 配置项 | 键名 | 默认值 | 说明 |", "|---|---|---|---|"]
    for name, field in plugin_module.SleepConfig.model_fields.items():
        title = field.title or name
        rows.append(
            f"| {title} | `{name}` | {_render_default(field.default)} | {_describe(field)} |"
        )
    return "\n".join(rows)


def current_readme_table(readme: str) -> str | None:
    if BEGIN not in readme or END not in readme:
        return None
    return readme.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def write(readme_path: Path) -> bool:
    readme = readme_path.read_text(encoding="utf-8")
    if BEGIN not in readme or END not in readme:
        raise SystemExit(f"markers {BEGIN} / {END} not found in {readme_path}")
    head, rest = readme.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    updated = f"{head}{BEGIN}\n{build_table()}\n{END}{tail}"
    if updated == readme:
        return False
    readme_path.write_text(updated, encoding="utf-8")
    return True


if __name__ == "__main__":
    if "--write" in sys.argv:
        changed = write(ROOT / "README.md")
        print("README updated" if changed else "README already up to date")
    else:
        print(build_table())
