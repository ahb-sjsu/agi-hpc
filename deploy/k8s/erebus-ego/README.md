# Erebus Ego — self-hosted vLLM probe

Kubernetes manifests for the Erebus ego fine-tuning pod on NRP Nautilus, namespace `ssu-atlas-ai`. See `project_erebus_ego_architecture` memory for the architectural rationale.

## What's here

| File | Purpose |
|---|---|
| `probe-glm-air.yaml` | One-shot probe pod: vLLM + GLM-4.5-Air-AWQ on 4× A10. Validates the stack before persistent deployment. |
| `service.yaml` | ClusterIP Service fronting the pod on port 8000. Consumed by the Atlas chat handler cascade. |
| `README.md` | this file |

## Apply

```bash
# From a workstation with kubectl configured against NRP
kubectl apply -f deploy/k8s/erebus-ego/probe-glm-air.yaml
kubectl apply -f deploy/k8s/erebus-ego/service.yaml

# Watch
kubectl -n ssu-atlas-ai get pods -l app=erebus-ego -w

# Exec in for debugging
kubectl -n ssu-atlas-ai exec -it deploy/erebus-ego-probe -- bash
```

## Sanity probe

Once the pod reports `Ready 1/1`, run the bundled probe:

```bash
python scripts/probe_erebus_ego.py \
  --url http://erebus-ego.ssu-atlas-ai.svc.cluster.local:8000
```

From Atlas (leaf-link-connected), the same script works via the cluster DNS or a `kubectl port-forward`.

## Policy context

This pod is the one exception to the "no ML serving on NRP GPU pods" rule — justified only if it sustains >40% utilization. The justification path is: all Erebus chats + Primer fallback + Council fallback route here, keeping the pod busy enough to meet the utilization threshold. If measured utilization stays low during the probe, we fold the manifest back and run the ego from managed `glm-4.7` (NRP ellm) instead of self-hosting.

The PVC `erebus-ego-models` (300 Gi, `rook-cephfs`, Bound) is pre-provisioned and holds the AWQ weights across pod restarts. Do not delete it without reprovisioning the weights.

L40 nodes are effectively reserved for `csu-tide` — see `reference_nrp_l40_reservation`. These manifests target A10 nodes, where ssu-atlas-ai has usable quota.
