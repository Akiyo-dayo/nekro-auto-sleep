"""The README config table is generated; this fails when it drifts from the code.

The table was hand-written until a third of the fields were missing from it and
two documented defaults were stale. Deriving it from `SleepConfig` means the
docs cannot quietly disagree with the plugin any more.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.gen_config_table import build_table, current_readme_table

README = Path(__file__).resolve().parent.parent / "README.md"


def test_readme_config_table_matches_the_config_class():
    table = current_readme_table(README.read_text(encoding="utf-8"))
    assert table is not None, "config-table markers missing from README.md"
    assert table == build_table(), (
        "README config table is stale; regenerate with "
        "`python tools/gen_config_table.py --write`"
    )


def test_every_config_field_is_documented():
    import nekro_auto_sleep as plugin_module

    table = current_readme_table(README.read_text(encoding="utf-8")) or ""
    missing = [
        name
        for name in plugin_module.SleepConfig.model_fields
        if f"`{name}`" not in table
    ]
    assert not missing, f"undocumented config fields: {missing}"


@pytest.mark.parametrize(
    "section",
    [
        "## 从 v1 升级",
        "## 运维指令",
        "## 作息覆盖",
        "### 多实例共群",
        "## 拟人化细节",
        "## 开发",
        "## 已知未完项",
    ],
)
def test_readme_keeps_its_sections(section):
    assert section in README.read_text(encoding="utf-8")
