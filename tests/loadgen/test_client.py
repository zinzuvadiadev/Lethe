import httpx
import respx

from loadgen.client import send_completion_request

SSE_BODY = (
    b'data: {"choices":[{"text":"Hello"}]}\n\n'
    b'data: {"choices":[{"text":" world"}]}\n\n'
    b'data: [DONE]\n\n'
)


@respx.mock
async def test_send_completion_request_counts_tokens_and_records_ttft():
    respx.post("http://localhost:8000/v1/completions").mock(
        return_value=httpx.Response(
            200, content=SSE_BODY, headers={"content-type": "text/event-stream"}
        )
    )
    async with httpx.AsyncClient() as client:
        result = await send_completion_request(
            client,
            base_url="http://localhost:8000",
            served_model_name="qwen3-4b-instruct-2507",
            prompt="hi",
            context_tokens=10,
            max_tokens=50,
            timeout_sec=5.0,
        )
    assert result.success is True
    assert result.completed_output_tokens == 2
    assert result.ttft_sec is not None
    assert result.latency_sec >= result.ttft_sec
    assert result.error is None


@respx.mock
async def test_send_completion_request_records_failure_on_http_error():
    respx.post("http://localhost:8000/v1/completions").mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient() as client:
        result = await send_completion_request(
            client,
            base_url="http://localhost:8000",
            served_model_name="qwen3-4b-instruct-2507",
            prompt="hi",
            context_tokens=10,
            max_tokens=50,
            timeout_sec=5.0,
        )
    assert result.success is False
    assert result.completed_output_tokens == 0
    assert result.ttft_sec is None
    assert result.error is not None
