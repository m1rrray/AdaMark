"""Checkpoint IO and the ablation metadata stored alongside the weights

The three ablation switches change the embedding path, not just the loss, so
weights are only meaningful together with the switches they were trained under.
Every file written here carries them, and loading validates them.
"""

import torch

# Switches that change the embedding path and must match at inference time.
EMBEDDING_SWITCHES = ("use_film", "use_budget")

# Switches recorded for provenance only; they affect training, not inference.
TRAINING_SWITCHES = ("use_fft_loss",)

ABLATION_KEYS = EMBEDDING_SWITCHES + TRAINING_SWITCHES


def ablation_from_config(config):
    """Read the ablation switches out of a train or eval config"""

    return {k: bool(config.get(k, True)) for k in ABLATION_KEYS}


def save_checkpoint(path, epoch, hiding_net, revealing_net,
                    optimizer_h, optimizer_r, scheduler_h, scheduler_r,
                    scaler_h, best_val_loss, epochs_no_improve, best_epoch,
                    clean_ref, jpeg_ref, blur_ref, ablation):
    """Save full training state, whose keys are part of the checkpoint format"""

    ckpt = {
        "epoch": epoch,
        "hiding_net": hiding_net.module.state_dict(),
        "revealing_net": revealing_net.module.state_dict(),
        "optimizer_h": optimizer_h.state_dict(),
        "optimizer_r": optimizer_r.state_dict(),
        "scheduler_h": scheduler_h.state_dict(),
        "scheduler_r": scheduler_r.state_dict(),
        "scaler_h": scaler_h.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
        "best_epoch": best_epoch,
        "clean_ref": clean_ref,
        "jpeg_ref": jpeg_ref,
        "blur_ref": blur_ref,
        "ablation": dict(ablation),
    }
    torch.save(ckpt, path)


def save_model(path, state_dict, ablation, epoch=None):
    """Save a single network's weights together with its ablation metadata"""

    torch.save({
        "state_dict": state_dict,
        "ablation": dict(ablation),
        "epoch": epoch,
    }, path)


def validate_ablation(stored, expected, path):
    """Raise when stored and expected switches disagree on the embedding path"""

    if stored is None:
        print(f"[!] {path} carries no ablation metadata; cannot verify it matches "
              f"the current config.")
        return

    mismatched = [k for k in EMBEDDING_SWITCHES if stored.get(k) != expected.get(k)]
    if mismatched:
        detail = ", ".join(
            f"{k}: checkpoint={stored.get(k)} config={expected.get(k)}" for k in mismatched)
        raise ValueError(
            f"Ablation mismatch for {path}. The weights were trained with a different "
            f"embedding path, so evaluating them under the current config would be "
            f"meaningless ({detail}). Align the flags in your eval config."
        )

    for k in TRAINING_SWITCHES:
        if k in stored and stored.get(k) != expected.get(k):
            print(f"[i] {path}: trained with {k}={stored[k]}, config says {expected.get(k)}. "
                  f"This affects training only.")


def load_model_state(path, map_location, ablation, weights_key=None):
    """Load a state dict, validating the ablation metadata when the file carries it

    Accepts three layouts: a file written by ``save_model``, a full training
    checkpoint when ``weights_key`` names the entry to pull out, and a bare state
    dict from before the metadata existed.
    """

    obj = torch.load(path, map_location=map_location)

    if not isinstance(obj, dict):
        return obj

    stored_ablation = obj.get("ablation")

    if weights_key is not None and weights_key in obj:
        state_dict = obj[weights_key]
    elif "state_dict" in obj:
        state_dict = obj["state_dict"]
    else:
        state_dict = obj

    validate_ablation(stored_ablation, ablation, path)

    return state_dict
