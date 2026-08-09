import httpx
import pytest
import respx

from loadgen.server_lifecycle import ensure_port_free, wait_for_server_ready


@respx.mock
async def test_wait_for_server_ready_returns_true_on_first_healthy_response():
    respx.get("http://localhost:8000/health").mock(return_value=httpx.Response(200))
    result = await wait_for_server_ready("http://localhost:8000", timeout_sec=5.0, poll_interval_sec=0.01)
    assert result is True


@respx.mock
async def test_wait_for_server_ready_times_out_if_never_healthy():
    respx.get("http://localhost:8000/health").mock(side_effect=httpx.ConnectError("refused"))
    result = await wait_for_server_ready("http://localhost:8000", timeout_sec=0.05, poll_interval_sec=0.01)
    assert result is False


@respx.mock
async def test_wait_for_server_ready_recovers_after_transient_failures():
    route = respx.get("http://localhost:8000/health")
    route.side_effect = [
        httpx.ConnectError("refused"),
        httpx.ConnectError("refused"),
        httpx.Response(200),
    ]
    result = await wait_for_server_ready("http://localhost:8000", timeout_sec=5.0, poll_interval_sec=0.01)
    assert result is True


@respx.mock
async def test_ensure_port_free_passes_when_nothing_listening():
    respx.get("http://localhost:8000/health").mock(side_effect=httpx.ConnectError("refused"))
    await ensure_port_free("http://localhost:8000")  # must not raise


@respx.mock
async def test_ensure_port_free_raises_when_something_healthy():
    respx.get("http://localhost:8000/health").mock(return_value=httpx.Response(200))
    with pytest.raises(RuntimeError, match="already responding"):
        await ensure_port_free("http://localhost:8000")
