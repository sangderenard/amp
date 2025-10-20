import pytest

from amp import c_kernels


def test_c_kernel_reports_reason_when_unavailable():
    if c_kernels.AVAILABLE:
        pytest.skip("C kernels available; diagnostic not applicable")
    assert c_kernels.UNAVAILABLE_REASON, "Expected descriptive reason for missing C kernels"
