"""Distributed DDP training loop for the adaptive watermarking model"""

import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from kornia.metrics import psnr, ssim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from adamark.attacks.attack_module import AttackModule
from adamark.attacks.distortion import distortion_deterministic_oneof
from adamark.data.dataset import ImageDataset
from adamark.data.transforms import get_adaptive_transforms
from adamark.losses import FFTLoss, HidingLoss, RevealingLoss, enforce_watermark_budget
from adamark.models import HidingNet, RevealingNet
from adamark.training.checkpoint import save_checkpoint
from adamark.utils import ddp_mean, generate_qr_payloads, set_seed

logger = logging.getLogger(__name__)


def train(config):
    # ----- Initialization -----
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    seed = int(config.get("seed", 42))
    set_seed(seed)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    if rank == 0:
        logger.info("DDP started: world_size=%d, visible_gpus=%d",
                    world_size, torch.cuda.device_count())

    use_film = bool(config.get("use_film", True))
    use_fft_loss = bool(config.get("use_fft_loss", True))
    use_budget = bool(config.get("use_budget", True))

    hiding_net = HidingNet(film_scale=1.0, use_film=use_film).to(device)
    revealing_net = RevealingNet().to(device)
    attack_module = AttackModule().to(device)

    hiding_net = DDP(hiding_net, device_ids=[local_rank])
    revealing_net = DDP(revealing_net, device_ids=[local_rank])

    optimizer_h = torch.optim.Adam(hiding_net.parameters(), lr=config["learning_rate"])
    optimizer_r = torch.optim.Adam(revealing_net.parameters(), lr=config["learning_rate"])
    eta_min = float(config.get("eta_min", 1e-7))
    T_max = int(config.get("T_max", config["epochs"]))

    scheduler_h = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_h, T_max=T_max, eta_min=eta_min)
    scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_r, T_max=T_max, eta_min=eta_min)

    hiding_criterion = HidingLoss()
    revealing_criterion = RevealingLoss()
    fft_loss_fn = FFTLoss()

    transform = get_adaptive_transforms()
    full_dataset = ImageDataset(cover_dir=config["train_cover_dir"], transform=transform)

    n = len(full_dataset)
    val_ratio = float(config.get("val_ratio", 0.1))
    train_size = int((1.0 - val_ratio) * n)

    split_seed = int(config.get("split_seed", seed))
    g_split = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n, generator=g_split).tolist()
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                       shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                     shuffle=False, drop_last=False)

    num_workers = int(config.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn,
        persistent_workers=True, prefetch_factor=3,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], sampler=val_sampler,
        num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn,
        persistent_workers=True, prefetch_factor=3,
    )

    save_dir = os.path.abspath(config.get("save_dir", "checkpoints"))
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        logger.info("save_dir: %s | cwd: %s", save_dir, os.getcwd())
    dist.barrier()

    patience = int(config.get("patience", 3))
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    clean_ref = None
    jpeg_ref = None
    blur_ref = None

    scaler_h = torch.cuda.amp.GradScaler()

    lambda_r = config.get("revealing_loss_weight", 1.0)
    lambda_fft = config.get("lambda_fft", 0.5)

    if rank == 0 and not (use_film and use_fft_loss and use_budget):
        logger.info("[ABLATION] use_film=%s use_fft_loss=%s use_budget=%s",
                    use_film, use_fft_loss, use_budget)

    resume = bool(config.get("resume", False))
    resume_path = config.get("resume_path", None)
    start_epoch = 0

    # ----- Resume from checkpoint -----
    if resume and resume_path is not None and os.path.isfile(resume_path):
        map_location = {"cuda:0": f"cuda:{local_rank}"}
        ckpt = torch.load(resume_path, map_location=map_location)

        hiding_net.module.load_state_dict(ckpt["hiding_net"], strict=True)
        revealing_net.module.load_state_dict(ckpt["revealing_net"], strict=True)

        warm_start = bool(config.get("warm_start", False))

        if not warm_start:
            if "optimizer_h" in ckpt:
                optimizer_h.load_state_dict(ckpt["optimizer_h"])
            if "optimizer_r" in ckpt:
                optimizer_r.load_state_dict(ckpt["optimizer_r"])
            if "scheduler_h" in ckpt:
                scheduler_h.load_state_dict(ckpt["scheduler_h"])
            if "scheduler_r" in ckpt:
                scheduler_r.load_state_dict(ckpt["scheduler_r"])
            if "scaler_h" in ckpt:
                scaler_h.load_state_dict(ckpt["scaler_h"])
        else:
            if rank == 0:
                logger.info("[WARM START] Ignoring optimizer states. Starting at LR=%s",
                            config["learning_rate"])

            start_epoch = int(ckpt.get("epoch", -1)) + 1
            remaining_epochs = int(config["epochs"]) - start_epoch

            scheduler_h = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_h, T_max=remaining_epochs, eta_min=eta_min)
            scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_r, T_max=remaining_epochs, eta_min=eta_min)

        clean_ref = ckpt.get("clean_ref", None)
        jpeg_ref = ckpt.get("jpeg_ref", None)
        blur_ref = ckpt.get("blur_ref", None)

        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", epochs_no_improve)
        best_epoch = ckpt.get("best_epoch", best_epoch)
        start_epoch = int(ckpt.get("epoch", -1)) + 1

        if rank == 0:
            logger.info("[RESUME] Weights loaded from: %s", resume_path)
            logger.info("[RESUME] start_epoch=%d, best_val_loss=%.6f", start_epoch, best_val_loss)
    else:
        if rank == 0 and resume:
            logger.warning("[RESUME] resume=True but checkpoint not found: %s", resume_path)

    if rank == 0:
        logger.info("--- Training started ---")

    # ----- Training loop -----
    for epoch in range(start_epoch, int(config["epochs"])):
        train_sampler.set_epoch(epoch)

        hiding_net.train()
        revealing_net.train()

        train_sum = {
            "loss": 0.0, "loss_h": 0.0, "loss_r": 0.0, "loss_fft": 0.0,
            "psnr_h": 0.0, "ssim_h": 0.0, "psnr_r": 0.0, "ssim_r": 0.0,
        }
        train_n = 0

        for cover_images, _ in train_loader:
            cover_images = cover_images.to(device, non_blocking=True)
            B, C, H, W = cover_images.shape
            random_block_size = random.choice([8, 16])
            secret_images = generate_qr_payloads(B, H, W, block_size=random_block_size, device=device)

            # Sample rf: spike at 0 with prob p0, spike at 1 with prob p1, else uniform.
            p0, p1 = 0.2, 0.2
            u = torch.rand((B,), device=device)
            rf_tensor = torch.empty((B,), device=device)

            mask0 = (u < p0)
            mask1 = (u >= p0) & (u < p0 + p1)
            masku = (u >= p0 + p1)

            rf_tensor[mask0] = 0.0
            rf_tensor[mask1] = 1.0
            rf_tensor[masku] = torch.rand((masku.sum().item(),), device=device)

            rf_tensor = rf_tensor.unsqueeze(1)

            optimizer_h.zero_grad(set_to_none=True)
            optimizer_r.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                watermark = hiding_net(secret_images, rf_tensor)
                if use_budget:
                    watermark = enforce_watermark_budget(watermark, rf_tensor)

                container = (cover_images + watermark).clamp(0.0, 1.0)
                effective_watermark = container - cover_images

                tampered_container, mask = attack_module(
                    container=container, cover=cover_images, rf=rf_tensor)
                retrieved_secret = revealing_net(tampered_container)

                loss_h_batch = hiding_criterion(container, cover_images, ssim_func=ssim)
                loss_r_batch = revealing_criterion(
                    retrieved_secret, secret_images, mask, lambda_r_base=lambda_r)
                loss_fft = (fft_loss_fn(effective_watermark, rf_tensor) if use_fft_loss
                            else effective_watermark.new_zeros(()))

            total_loss_val = loss_h_batch + loss_r_batch + lambda_fft * loss_fft

            scaler_h.scale(total_loss_val).backward()

            scaler_h.unscale_(optimizer_h)
            scaler_h.unscale_(optimizer_r)
            torch.nn.utils.clip_grad_norm_(hiding_net.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(revealing_net.parameters(), max_norm=1.0)

            scaler_h.step(optimizer_h)
            scaler_h.step(optimizer_r)
            scaler_h.update()

            with torch.no_grad():
                train_sum["loss"] += total_loss_val.detach().item()
                train_sum["loss_h"] += loss_h_batch.detach().item()
                train_sum["loss_r"] += loss_r_batch.detach().item()
                train_sum["loss_fft"] += loss_fft.detach().item()
                train_sum["psnr_h"] += psnr(container.float(), cover_images.float(), max_val=1.0).item()
                train_sum["ssim_h"] += ssim(container.float(), cover_images.float(),
                                            window_size=11, max_val=1.0).mean().item()
                train_sum["psnr_r"] += psnr(retrieved_secret.float(), secret_images.float(), max_val=1.0).item()
                train_sum["ssim_r"] += ssim(retrieved_secret.float(), secret_images.float(),
                                            window_size=11, max_val=1.0).mean().item()
                train_n += 1

        train_avg_local = {k: train_sum[k] / max(train_n, 1) for k in train_sum}
        train_avg = {k: ddp_mean(v, device, world_size) for k, v in train_avg_local.items()}

        # ----- Validation loop -----
        hiding_net.eval()
        revealing_net.eval()

        val_tamper = attack_module.tampering

        val_clean_sum = {"loss": 0.0, "psnr_h": 0.0, "ssim_h": 0.0, "psnr_r": 0.0, "ssim_r": 0.0, "fft": 0.0}
        val_jpeg_sum = {"loss": 0.0, "psnr_h": 0.0, "ssim_h": 0.0, "psnr_r": 0.0, "ssim_r": 0.0, "fft": 0.0}
        val_blur_sum = {"loss": 0.0, "psnr_h": 0.0, "ssim_h": 0.0, "psnr_r": 0.0, "ssim_r": 0.0, "fft": 0.0}
        val_n = 0

        with torch.no_grad():
            for cover_img, _ in val_loader:
                cover_img = cover_img.to(device, non_blocking=True)
                B, _, H, W = cover_img.shape

                secret_img = generate_qr_payloads(B, H, W, block_size=8, device=device)

                with torch.cuda.amp.autocast():
                    rf_val = torch.ones((B, 1), device=device)
                    watermark = hiding_net(secret_img, rf_val)
                    if use_budget:
                        watermark = enforce_watermark_budget(watermark, rf_val)

                    container = (cover_img + watermark).clamp(0.0, 1.0)
                    effective_watermark = container - cover_img

                    loss_h = hiding_criterion(container, cover_img, ssim_func=ssim)
                    loss_fft = (fft_loss_fn(effective_watermark, rf_val) if use_fft_loss
                                else effective_watermark.new_zeros(()))

                    tamp_container, mask = val_tamper(container, cover_img)

                    # Keep half of the batch clean.
                    half_B = B // 2
                    if half_B > 0:
                        tamp_container[:half_B] = container[:half_B]
                        mask[:half_B] = 0.0

                    # Clean branch.
                    retrieved_clean = revealing_net(tamp_container)
                    loss_r_clean = revealing_criterion(retrieved_clean, secret_img, mask,
                                                       lambda_r_base=lambda_r)
                    total_clean = loss_h + loss_r_clean

                    # JPEG branch.
                    dist_jpeg = distortion_deterministic_oneof(
                        tamp_container, rf_attack=1.0, mode="jpeg", min_jpeg_quality=30.0)
                    retrieved_jpeg = revealing_net(dist_jpeg)
                    loss_r_jpeg = revealing_criterion(retrieved_jpeg, secret_img, mask,
                                                      lambda_r_base=lambda_r)
                    total_jpeg = loss_h + loss_r_jpeg

                    # Blur branch.
                    dist_blur = distortion_deterministic_oneof(
                        tamp_container, rf_attack=1.0, mode="blur", max_blur_sigma=1.5,
                        blur_kernel=(11, 11))
                    retrieved_blur = revealing_net(dist_blur)
                    loss_r_blur = revealing_criterion(retrieved_blur, secret_img, mask,
                                                      lambda_r_base=lambda_r)
                    total_blur = loss_h + loss_r_blur

                psnr_h_val = psnr(container.float(), cover_img.float(), max_val=1.0).item()
                ssim_h_val = ssim(container.float(), cover_img.float(),
                                  window_size=11, max_val=1.0).mean().item()
                fft_val = float(loss_fft.item())

                val_clean_sum["loss"] += total_clean.item()
                val_clean_sum["psnr_h"] += psnr_h_val
                val_clean_sum["ssim_h"] += ssim_h_val
                val_clean_sum["psnr_r"] += psnr(retrieved_clean.float(), secret_img.float(), max_val=1.0).item()
                val_clean_sum["ssim_r"] += ssim(retrieved_clean.float(), secret_img.float(),
                                                window_size=11, max_val=1.0).mean().item()
                val_clean_sum["fft"] += fft_val

                val_jpeg_sum["loss"] += total_jpeg.item()
                val_jpeg_sum["psnr_h"] += psnr_h_val
                val_jpeg_sum["ssim_h"] += ssim_h_val
                val_jpeg_sum["psnr_r"] += psnr(retrieved_jpeg.float(), secret_img.float(), max_val=1.0).item()
                val_jpeg_sum["ssim_r"] += ssim(retrieved_jpeg.float(), secret_img.float(),
                                               window_size=11, max_val=1.0).mean().item()
                val_jpeg_sum["fft"] += fft_val

                val_blur_sum["loss"] += total_blur.item()
                val_blur_sum["psnr_h"] += psnr_h_val
                val_blur_sum["ssim_h"] += ssim_h_val
                val_blur_sum["psnr_r"] += psnr(retrieved_blur.float(), secret_img.float(), max_val=1.0).item()
                val_blur_sum["ssim_r"] += ssim(retrieved_blur.float(), secret_img.float(),
                                               window_size=11, max_val=1.0).mean().item()
                val_blur_sum["fft"] += fft_val

                val_n += 1

        val_clean_avg_local = {k: val_clean_sum[k] / max(val_n, 1) for k in val_clean_sum}
        val_jpeg_avg_local = {k: val_jpeg_sum[k] / max(val_n, 1) for k in val_jpeg_sum}
        val_blur_avg_local = {k: val_blur_sum[k] / max(val_n, 1) for k in val_blur_sum}

        val_clean_avg = {k: ddp_mean(v, device, world_size) for k, v in val_clean_avg_local.items()}
        val_jpeg_avg = {k: ddp_mean(v, device, world_size) for k, v in val_jpeg_avg_local.items()}
        val_blur_avg = {k: ddp_mean(v, device, world_size) for k, v in val_blur_avg_local.items()}

        stop = torch.tensor([0], device=device, dtype=torch.int32)

        if rank == 0:
            logger.info("--- Epoch [%d/%s] ---", epoch + 1, config["epochs"])
            logger.info("Train -> Total: %.4f | H: %.4f | R: %.4f | FFT: %.4f",
                        train_avg["loss"], train_avg["loss_h"], train_avg["loss_r"], train_avg["loss_fft"])
            logger.info("  Train -> Loss: %.4f | PSNR-H: %.2f | SSIM-H: %.3f | PSNR-R: %.2f | SSIM-R: %.3f",
                        train_avg["loss"], train_avg["psnr_h"], train_avg["ssim_h"],
                        train_avg["psnr_r"], train_avg["ssim_r"])
            logger.info("  Val(clean) -> Loss: %.4f | PSNR-H: %.2f | SSIM-H: %.3f | PSNR-R: %.2f | SSIM-R: %.3f",
                        val_clean_avg["loss"], val_clean_avg["psnr_h"], val_clean_avg["ssim_h"],
                        val_clean_avg["psnr_r"], val_clean_avg["ssim_r"])
            logger.info("  Val(JPEG)  -> Loss: %.4f | PSNR-H: %.2f | SSIM-H: %.3f | PSNR-R: %.2f | SSIM-R: %.3f",
                        val_jpeg_avg["loss"], val_jpeg_avg["psnr_h"], val_jpeg_avg["ssim_h"],
                        val_jpeg_avg["psnr_r"], val_jpeg_avg["ssim_r"])
            logger.info("  Val(Blur)  -> Loss: %.4f | PSNR-H: %.2f | SSIM-H: %.3f | PSNR-R: %.2f | SSIM-R: %.3f",
                        val_blur_avg["loss"], val_blur_avg["psnr_h"], val_blur_avg["ssim_h"],
                        val_blur_avg["psnr_r"], val_blur_avg["ssim_r"])
            logger.info("  FFT -> clean: %.6f | JPEG: %.6f | Blur: %.6f",
                        val_clean_avg["fft"], val_jpeg_avg["fft"], val_blur_avg["fft"])

            if clean_ref is None:
                clean_ref = max(val_clean_avg["loss"], 1e-8)
            if jpeg_ref is None:
                jpeg_ref = max(val_jpeg_avg["loss"], 1e-8)
            if blur_ref is None:
                blur_ref = max(val_blur_avg["loss"], 1e-8)

            clean_norm = val_clean_avg["loss"] / clean_ref
            jpeg_norm = val_jpeg_avg["loss"] / jpeg_ref
            blur_norm = val_blur_avg["loss"] / blur_ref

            # Selection score based on normalized losses across attacks.
            robust_norm = max(jpeg_norm, blur_norm)
            avg_val_loss = 0.5 * clean_norm + 0.5 * robust_norm

            logger.info("  Val score -> clean: %.4f | jpeg: %.4f | blur: %.4f | robust(worst): %.4f | total: %.4f",
                        clean_norm, jpeg_norm, blur_norm, robust_norm, avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_epoch = epoch + 1

                torch.save(hiding_net.module.state_dict(), os.path.join(save_dir, "hiding_net_best.pth"))
                torch.save(revealing_net.module.state_dict(), os.path.join(save_dir, "revealing_net_best.pth"))
                logger.info("  -> New best result, models saved")
            else:
                epochs_no_improve += 1
                logger.info("  -> No improvement, early-stop counter: %d/%d", epochs_no_improve, patience)

            save_checkpoint(
                os.path.join(save_dir, "last.pth"),
                epoch=epoch,
                hiding_net=hiding_net,
                revealing_net=revealing_net,
                optimizer_h=optimizer_h,
                optimizer_r=optimizer_r,
                scheduler_h=scheduler_h,
                scheduler_r=scheduler_r,
                scaler_h=scaler_h,
                best_val_loss=best_val_loss,
                epochs_no_improve=epochs_no_improve,
                best_epoch=best_epoch,
                clean_ref=clean_ref,
                jpeg_ref=jpeg_ref,
                blur_ref=blur_ref,
            )

            if epochs_no_improve >= patience:
                logger.info("Early stopping: val loss did not improve for %d epochs", patience)
                logger.info("Best result was at epoch %d with val loss: %.4f", best_epoch, best_val_loss)
                stop[0] = 1

        dist.broadcast(stop, src=0)
        if stop.item() == 1:
            break

        scheduler_h.step()
        scheduler_r.step()

    if rank == 0:
        logger.info("--- Training finished ---")

    dist.destroy_process_group()
