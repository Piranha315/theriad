import time

import pytest

from theriad.gateway.contracts import ChatRequest, Deployment, EndpointPolicy, eligible
from theriad.gateway.vertex import vertex_deployment


def deployment(location):
    # Construction fixtures test the declared network boundary, not live model availability.
    return vertex_deployment(
        project="pilot-project",
        location=location,
        model="google/fixture-model",
        revision="fixture-revision",
        context_window=64000,
        max_output_tokens=8192,
        input_price=1,
        output_price=2,
        priced_at=time.time(),
        tiers=["cheap_code"],
    )


@pytest.mark.parametrize("location", ["us", "eu", "us-central1"])
def test_google_locations_require_exact_endpoint_and_explicit_policy(location):
    selected = deployment(location)
    request = ChatRequest(messages=[{"role": "user", "content": "hello"}])
    policy = EndpointPolicy(
        allowed_endpoints=[selected.endpoint],
        allowed_regions=[location],
    )
    assert eligible(selected, request, policy)
    other_location = "eu" if location != "eu" else "us"
    assert not eligible(
        selected, request, policy.model_copy(update={"allowed_regions": [other_location]})
    )
    assert not eligible(selected, request, policy.model_copy(update={"allowed_endpoints": []}))
    for endpoint in (
        selected.endpoint.replace(
            selected.endpoint.split("/")[2], f"aiplatform.{other_location}.rep.googleapis.com"
        ),
        selected.endpoint.replace(f"/locations/{location}/", f"/locations/{other_location}/"),
        selected.endpoint.replace("/projects/pilot-project/", "/projects/other-project/"),
    ):
        with pytest.raises(ValueError, match="declared managed Vertex endpoint"):
            Deployment.model_validate({**selected.model_dump(), "endpoint": endpoint})


@pytest.mark.parametrize("location", ["global", "asia", "US", "us/other"])
def test_google_locations_never_enable_global_or_undeclared_boundary(location):
    with pytest.raises(ValueError):
        deployment(location)


def test_multi_region_is_not_an_individual_region_permission():
    selected = deployment("us")
    request = ChatRequest(messages=[{"role": "user", "content": "hello"}])
    policy = EndpointPolicy(allowed_endpoints=[selected.endpoint], allowed_regions=["us-central1"])
    assert not eligible(selected, request, policy)


@pytest.mark.parametrize("location", ["us", "eu"])
def test_multi_region_rejects_the_invalid_regional_hostname(location):
    selected = deployment(location)
    assert selected.endpoint.startswith(f"https://aiplatform.{location}.rep.googleapis.com/")
    with pytest.raises(ValueError, match="declared managed Vertex endpoint"):
        Deployment.model_validate(
            {
                **selected.model_dump(),
                "endpoint": selected.endpoint.replace(
                    f"aiplatform.{location}.rep.googleapis.com",
                    f"{location}-aiplatform.googleapis.com",
                ),
            }
        )
