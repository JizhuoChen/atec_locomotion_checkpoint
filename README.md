# ATEC B2-Piper locomotion checkpoint snapshot

This repository preserves the training framework that produced the canonical August 7 B2-Piper rough-terrain checkpoint:

`model_19998.pt` — SHA-256 `995b9d11ae99648255e5c213baaa14cfbe31b0380f76e976a3469a593322bfd9`

The detailed scene, observation, action, PPO, reward, curriculum and domain-randomization contracts are documented in [CHECKPOINT_SNAPSHOT.md](CHECKPOINT_SNAPSHOT.md).

## Included model artifacts

| Stage | Artifact | SHA-256 |
|---|---|---|
| Flat robust pretraining | `reference/lineage/flat/model_7999.pt` | `8d9d184d94cf707c45550c715ef5486547976cde2149d879cba9cf6aaedd8264` |
| Heading-rough actor-transfer training | `reference/lineage/heading_rough/model_11999.pt` | `c7a8b3b0e990c7fb724ea4408c0c2decfee8cfec5e7c0d9938d88bfa416813b8` |
| Fine-grained full-state continuation | `reference/model_19998_run/model_19998.pt` | `995b9d11ae99648255e5c213baaa14cfbe31b0380f76e976a3469a593322bfd9` |

The final TorchScript and ONNX policy exports are in `reference/model_19998_run/exported/`. Resolved YAML configurations, TensorBoard events, spawn audits and original pipeline manifests are preserved beside the models.

## Assets

The upstream competition repository deliberately ignores the extracted `atec_robot_model` tree. It is therefore not committed here. Before training from a fresh clone, copy or link that directory from your competition checkout:

```bash
cd /path/to/atec_locomotion_checkpoint
cp -a /path/to/Clear_ATEC2026_Simulation_Challenge/atec_robot_model ./
```

The expected hashes for every asset file are recorded in `reference/provenance/atec_robot_model.sha256`.

## Verify and continue training

```bash
cd /path/to/atec_locomotion_checkpoint

/home/user/miniforge3/envs/isaaclab/bin/python \
  verify_checkpoint_snapshot.py --all-assets

./train_model_19998.sh --device cuda:0
```

The launcher restores the complete PPO state from `model_11999.pt`, trains 2,048 environments for another 8,000 updates with seed 42, and produces iteration 19,998. It also prevents the conda environment from silently importing a different editable checkout.

Exact checkpoint bytes are provided, but stochastic retraining is not expected to recreate identical output bytes because Isaac environment state and all GPU/PhysX RNG state were not stored in the source checkpoint.
