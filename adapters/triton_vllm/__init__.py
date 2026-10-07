"""Real HTTP clients for NVIDIA Triton Inference Server and vLLM.

IMPORTANT — honesty note (see README.md "Status"):
No real Triton Inference Server or vLLM server was deployed/running in the
session that built this repo. These clients implement the real, documented
wire protocols (Triton's KServe v2 HTTP/REST inference API, and vLLM's
OpenAI-compatible HTTP API) and are exercised in tests against a local
in-process mock server (tests/test_adapters.py) — not against a live Triton
or vLLM backend. They are real, runnable code, just not end-to-end verified
against the real systems in this environment.
"""
