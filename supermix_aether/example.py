"""Standalone runnable demonstration of SUPERMIX-AETHER routing, Torch specialist, and external adapter gating."""

from __future__ import annotations

import json

from supermix_aether.interfaces import OrchestrationRouterProtocol, SpecialistModelCallableProtocol
from supermix_aether.router import (
    ExternalHTTPModelAdapter,
    SpecialistModel,
    SupermixAether,
    TorchNeuralSpecialist,
)


def main() -> None:
    router = SupermixAether(allow_external_apis=False)
    assert isinstance(router, OrchestrationRouterProtocol)

    torch_spec = TorchNeuralSpecialist()
    http_adapter = ExternalHTTPModelAdapter(endpoint_url="http://127.0.0.1:9999/mock")
    assert isinstance(torch_spec, SpecialistModelCallableProtocol)
    assert isinstance(http_adapter, SpecialistModelCallableProtocol)

    ext_model = SpecialistModel(
        model_id="external_mock",
        task_types=["research_synthesis"],
        backend_type="external_optional",
        handler=http_adapter,
        priority=10,
    )
    router.register_model(ext_model)

    torch_out = torch_spec({"features": [0.2, 0.8, 0.1, 0.5, 0.3, 0.9, 0.4, 0.6]})
    route_out = router.route_task("decision_arbitration", {"features": [0.4, 0.7, 0.2, 0.9, 0.1, 0.3, 0.5, 0.6]})

    print("=== SUPERMIX-AETHER ROUTER EXAMPLE ===")
    print(
        json.dumps(
            {
                "torch_backend": torch_out.get("backend"),
                "torch_scores_len": len(torch_out.get("action_scores", [])),
                "external_model_healthy": router.models["external_mock"].healthy,
                "routed_model": route_out.get("model_id"),
                "registered_models": len(router.get_status()["models"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
