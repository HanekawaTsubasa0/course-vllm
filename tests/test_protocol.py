from course_vllm.server.protocol import GenerateRequest


def test_generate_request_defaults():
    request = GenerateRequest(prompt="hello")
    assert request.stream is False
    assert request.sampling_params.max_tokens is None
