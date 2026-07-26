"""Checkpoint save helper for the training loop"""

import torch


def save_checkpoint(path, epoch, hiding_net, revealing_net,
                    optimizer_h, optimizer_r, scheduler_h, scheduler_r,
                    scaler_h, best_val_loss, epochs_no_improve, best_epoch,
                    clean_ref, jpeg_ref, blur_ref):
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
    }
    torch.save(ckpt, path)
