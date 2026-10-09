from theriad.core.providers.factory import build_provider, register_provider_adapter
from theriad.core.providers.interfaces import ProviderProtocol


class ExampleProvider(ProviderProtocol):
    def __init__(self, api_key: str, timeout: float = 30.0):
        self.api_key = api_key
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "example"

    async def complete(self, request):  # pragma: no cover - construction test only
        raise NotImplementedError

    async def health_check(self) -> bool:
        return True


def test_registered_provider_can_be_built_without_factory_dispatch_changes():
    register_provider_adapter(
        "example-test",
        lambda **kwargs: ExampleProvider(**kwargs),
        env_key="EXAMPLE_TEST_API_KEY",
        replace=True,
    )
    provider = build_provider("example-test", api_key="secret", timeout=7.0)
    assert provider.name == "example"
    assert provider.timeout == 7.0
