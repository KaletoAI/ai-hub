"""The suite never does DNS: `adapters._resolve_asset_host` (the address check before a
cloud artifact download) answers "does not resolve" for every fake vendor host the
MockTransport tests use — an unresolvable name is let through, so those tests behave
as before. The tests of the check itself (test_cloudtask.ArtifactUrlFence) patch it."""
import adapters as _adapters


async def _no_dns(host, port):
    raise OSError("tests do not resolve names")

_adapters._resolve_asset_host = _no_dns
