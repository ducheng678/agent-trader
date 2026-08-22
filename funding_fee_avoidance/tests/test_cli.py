from __future__ import annotations

import pytest

from funding_fee_avoidance.__main__ import main


def test_execute_requires_watch_to_prevent_unmanaged_one_shot_hedge():
    with pytest.raises(SystemExit) as exc:
        main(["--execute"])

    assert exc.value.code == 2
