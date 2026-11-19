"""Searchable CNN architecture space.

An architecture is a JSON-serialisable dict:

    {
      "resolution": 32,                # input resolution
      "activation": "hardswish",       # activation used everywhere
      "stem_channels": 16,
      "stages": [                      # 3 or 4 stages (variable depth)
        {"block": "mbconv", "channels": 24, "depth": 2, "kernel": 3, "expansion": 4, "stride": 1},
        ...
      ]
    }

Dimensions searched
-------------------
| dimension            | choices                                              |
|----------------------|------------------------------------------------------|
| input resolution     | 24, 28, 32                                           |
| activation           | relu, relu6, hardswish, silu, leaky_relu             |
| stem channels        | 8, 16, 24, 32                                        |
| number of stages     | 3, 4                                                 |
| blocks per stage     | 1, 2, 3                (conv depth: 3 - 12 blocks)   |
| channels per stage   | stage-specific lists (STAGE_CHANNELS)                |
| kernel size          | 3, 5, 7                (per stage)                   |
| block type           | conv | dwsep (depthwise separable) | mbconv          |
| expansion (mbconv)   | 2, 4, 6                                              |
"""
import random

RESOLUTIONS = [24, 28, 32]
ACTIVATIONS = ["relu", "relu6", "hardswish", "silu", "leaky_relu"]
STEM_CHANNELS = [8, 16, 24, 32]
NUM_STAGES = [3, 4]
BLOCK_TYPES = ["conv", "dwsep", "mbconv"]
KERNELS = [3, 5, 7]
DEPTHS = [1, 2, 3]
EXPANSIONS = [2, 4, 6]
STAGE_CHANNELS = [[16, 24, 32], [24, 32, 48, 64], [48, 64, 96, 128], [96, 128, 160, 192]]
STAGE_STRIDES = [1, 2, 2, 2]


def sample_config(choose) -> dict:
    """Build a config by calling choose(name, choices) for every decision.

    `choose` is `trial.suggest_categorical` for Optuna, or a random chooser.
    """
    cfg = {
        "resolution": choose("resolution", RESOLUTIONS),
        "activation": choose("activation", ACTIVATIONS),
        "stem_channels": choose("stem_channels", STEM_CHANNELS),
        "stages": [],
    }
    n = choose("num_stages", NUM_STAGES)
    for i in range(n):
        block = choose(f"s{i}_block", BLOCK_TYPES)
        stage = {
            "block": block,
            "channels": choose(f"s{i}_channels", STAGE_CHANNELS[i]),
            "depth": choose(f"s{i}_depth", DEPTHS),
            "kernel": choose(f"s{i}_kernel", KERNELS),
            "expansion": choose(f"s{i}_expansion", EXPANSIONS) if block == "mbconv" else 1,
            "stride": STAGE_STRIDES[i],
        }
        cfg["stages"].append(stage)
    return cfg


def random_config(rng: random.Random | None = None) -> dict:
    rng = rng or random.Random()
    return sample_config(lambda name, choices: rng.choice(choices))


def optuna_config(trial) -> dict:
    return sample_config(trial.suggest_categorical)


def describe(cfg: dict) -> str:
    """Compact one-line description, e.g. 'r32 hardswish stem16 | mb3x2e4@24 > ds5x1@48 > ...'."""
    short = {"conv": "cv", "dwsep": "ds", "mbconv": "mb"}
    parts = []
    for s in cfg["stages"]:
        e = f"e{s['expansion']}" if s["block"] == "mbconv" else ""
        parts.append(f"{short[s['block']]}{s['kernel']}x{s['depth']}{e}@{s['channels']}")
    return f"r{cfg['resolution']} {cfg['activation']} stem{cfg['stem_channels']} | " + " > ".join(parts)


def flatten(cfg: dict) -> dict:
    """Flat dict (one column per decision) for CSV export."""
    row = {"resolution": cfg["resolution"], "activation": cfg["activation"],
           "stem_channels": cfg["stem_channels"], "num_stages": len(cfg["stages"]),
           "total_blocks": sum(s["depth"] for s in cfg["stages"])}
    for i, s in enumerate(cfg["stages"]):
        for k in ("block", "channels", "depth", "kernel", "expansion"):
            row[f"s{i}_{k}"] = s[k]
    return row
