"""Root conftest: put the repo on sys.path and install the host stub.

This has to run before anything imports `nekro_auto_sleep`, because that
package imports the NekroAgent host at module level.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tests.hoststub import install_host_stub  # noqa: E402

install_host_stub()
