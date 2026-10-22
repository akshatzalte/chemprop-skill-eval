"""Describe a saved Chemprop model as JSON (run with the evaluation environment's Python).

usage: <env>/bin/python scripts/inspect_model.py MODEL [MODEL ...]

Accepts files written by `chemprop train` (best.pt) or `models.save_model` (.pt) and Lightning
checkpoints (.ckpt). Prints one JSON object per model with its class, message passing, aggregation,
and the (in, out) sizes of each predictor's linear layers. Graders use this to check architecture
claims, such as an FFN shape or an aggregation type, without trusting the agent's own report.
"""

import json
import sys

import torch

from chemprop import models


def linear_shapes(module) -> list[list[int]]:
    return [
        [m.in_features, m.out_features] for m in module.modules() if isinstance(m, torch.nn.Linear)
    ]


def load(path: str):
    errors = []
    for cls in (models.MPNN, models.MulticomponentMPNN, models.MolAtomBondMPNN):
        for loader in (cls.load_from_file, cls.load_from_checkpoint):
            try:
                return loader(path, map_location="cpu")
            except Exception as e:  # noqa: BLE001 - try the next loader
                errors.append(f"{cls.__name__}.{loader.__name__}: {type(e).__name__}")
    raise RuntimeError("; ".join(errors))


def describe(path: str) -> dict:
    model = load(path)
    info = {"path": path, "class": type(model).__name__}
    mp = model.message_passing
    info["message_passing"] = type(mp).__name__
    blocks = getattr(mp, "blocks", [mp])
    info["message_passing_input_dims"] = [b.W_i.in_features for b in blocks if hasattr(b, "W_i")]
    info["message_passing_hidden_dims"] = [b.W_h.out_features for b in blocks if hasattr(b, "W_h")]
    # (in, out) of each message-passing layer of the first block (W_i, W_h, W_o or W_vo, ...), so
    # graders can work out the atom and bond feature sizes
    info["message_passing_linear"] = {
        name: [m.in_features, m.out_features]
        for name, m in blocks[0].named_modules()
        if isinstance(m, torch.nn.Linear)
    }
    graph_transform = getattr(blocks[0], "graph_transform", None)
    for kind, attr in (("atom", "V_transform"), ("bond", "E_transform")):
        transform = getattr(graph_transform, attr, None)
        info[f"{kind}_features_scaled"] = transform is not None and not isinstance(
            transform, torch.nn.Identity
        )
    info["aggregation"] = (
        type(model.agg).__name__ if getattr(model, "agg", None) is not None else None
    )
    predictors = {}
    for name in ("predictor", "mol_predictor", "atom_predictor", "bond_predictor"):
        predictor = getattr(model, name, None)
        if predictor is not None:
            predictors[name] = {
                "class": type(predictor).__name__,
                "linear": linear_shapes(predictor),
            }
    info["predictors"] = predictors
    info["has_descriptor_transform"] = not isinstance(
        getattr(model, "X_d_transform", torch.nn.Identity()), torch.nn.Identity
    )
    return info


if __name__ == "__main__":
    for path in sys.argv[1:]:
        try:
            print(json.dumps(describe(path)))
        except Exception as e:  # noqa: BLE001 - report and continue with the next model
            print(json.dumps({"path": path, "error": f"{type(e).__name__}: {e}"[:500]}))
