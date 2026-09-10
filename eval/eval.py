from pathlib import Path
import sys
from utils.metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
)
from dataset.utils import Normalizer
from model import build_vae
from model.dit import build_dit
from utils.metrics import calculate_top_k, euclidean_distance_matrix
from utils import get_logger, get_timestamp
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataset import build_dataset
from utils import seed_everything
from collections import OrderedDict


@torch.no_grad()
def evaluate_matching_score(motion_loaders):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    logger.info("Evaluating MM Distance")
    text_list = {}
    dist_mat_list = {}
    length_list = {}
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        text_list[motion_loader_name] = []
        dist_mat_list[motion_loader_name] = []
        length_list[motion_loader_name] = []
        for batch in motion_loader:
            text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
            dist_mat = euclidean_distance_matrix(
                text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy()
            )
            text_list[motion_loader_name].append(batch[0])
            dist_mat_list[motion_loader_name].append(dist_mat)
            length_list[motion_loader_name].append(batch[3])
            mm_dist_sum += dist_mat.trace()
            argsmax = np.argsort(dist_mat, axis=1)

            top_k_mat = calculate_top_k(argsmax, top_k=3)
            top_k_count += top_k_mat.sum(axis=0)

            all_size += text_embeddings.shape[0]

            all_motion_embeddings.append(motion_embeddings.cpu().numpy())

        all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
        mm_dist = mm_dist_sum / all_size
        R_precision = top_k_count / all_size
        match_score_dict[motion_loader_name] = mm_dist
        R_precision_dict[motion_loader_name] = R_precision
        activation_dict[motion_loader_name] = all_motion_embeddings

        logger.info(f"---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}")
        R_precision_str = "; ".join(
            [f"(top {i + 1}): {R_precision[i]:.4f}" for i in range(len(R_precision))]
        )
        logger.info(f"---> [{motion_loader_name}] R_precision: {R_precision_str}")

    return match_score_dict, R_precision_dict, activation_dict


@torch.no_grad()
def evaluate_fid(groundtruth_loader, activation_dict):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    logger.info("Evaluating FID")
    for batch in groundtruth_loader:
        motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
        gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings, cfg.eval.emb_scale)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings, cfg.eval.emb_scale)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        logger.info(f"---> [{model_name}] FID: {fid:.4f}")
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict):
    eval_dict = OrderedDict({})
    logger.info("Evaluating Diversity")
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, cfg.eval.diversity_times, cfg.eval.emb_scale, cfg.eval.divide_by)
        eval_dict[model_name] = diversity
        logger.info(f"---> [{model_name}] Diversity: {diversity:.4f}")
    return eval_dict


@torch.no_grad()
def evaluate_multimodality(mm_motion_loaders):
    eval_dict = OrderedDict({})
    logger.info("Evaluating MultiModality")
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        for batch in mm_motion_loader:
            # (1, mm_replications, dim_pos)
            batch[1] = batch[1][0]
            batch[2] = batch[2][0]
            batch[3] = batch[3][0]
            motion_embedings = eval_wrapper.get_motion_embeddings(batch)
            mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, cfg.eval.mm_num_times)
        logger.info(f"---> [{model_name}] Multimodality: {multimodality:.4f}")
        eval_dict[model_name] = multimodality
    return eval_dict


def evaluation():
    all_metrics = OrderedDict(
        {
            "MM Distance": OrderedDict({}),
            "R_precision": OrderedDict({}),
            "FID": OrderedDict({}),
            "Diversity": OrderedDict({}),
            "MultiModality": OrderedDict({}),
        }
    )
    for replication in range(1, cfg.eval.replication_times + 1):
        logger.info(f"Replication {replication}/{cfg.eval.replication_times}")
        motion_loaders = {}
        mm_motion_loaders = {}
        motion_loaders["ground truth"] = gt_dataloader
        for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
            motion_loader, mm_motion_loader = motion_loader_getter()
            motion_loaders[motion_loader_name] = motion_loader
            mm_motion_loaders[motion_loader_name] = mm_motion_loader
        mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders)
        fid_score_dict = evaluate_fid(gt_dataloader, acti_dict)
        div_score_dict = evaluate_diversity(acti_dict)
        # mm_score_dict = evaluate_multimodality(mm_motion_loaders)

        for key, item in R_precision_dict.items():
            if key not in all_metrics["R_precision"]:
                all_metrics["R_precision"][key] = [item]
            else:
                all_metrics["R_precision"][key] += [item]

        for key, item in fid_score_dict.items():
            if key not in all_metrics["FID"]:
                all_metrics["FID"][key] = [item]
            else:
                all_metrics["FID"][key] += [item]

        for key, item in mat_score_dict.items():
            if key not in all_metrics["MM Distance"]:
                all_metrics["MM Distance"][key] = [item]
            else:
                all_metrics["MM Distance"][key] += [item]

        for key, item in div_score_dict.items():
            if key not in all_metrics["Diversity"]:
                all_metrics["Diversity"][key] = [item]
            else:
                all_metrics["Diversity"][key] += [item]

        # for key, item in mm_score_dict.items():
        #     if key not in all_metrics["MultiModality"]:
        #         all_metrics["MultiModality"][key] = [item]
        #     else:
        #         all_metrics["MultiModality"][key] += [item]

    for metric_name, metric_dict in all_metrics.items():
        logger.info(f"========== {metric_name} Summary ==========")
        for model_name, values in metric_dict.items():
            mean, conf_interval = get_metric_statistics(np.array(values))
            if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                logger.info(f"---> [{model_name}] Mean: {mean:.4f}, CInt: {conf_interval:.4f}")
            elif isinstance(mean, np.ndarray):
                msg = "; ".join(
                    [
                        f"(top {i + 1}): {mean[i]:.4f} CInt: {conf_interval[i]:.4f}"
                        for i in range(len(mean))
                    ]
                )
                logger.info(f"---> [{model_name}] {msg}")


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(cfg.eval.replication_times)
    return mean, conf_interval


if __name__ == "__main__":
    cli_cfg = OmegaConf.from_cli()
    if "exp_name" not in cli_cfg:
        raise ValueError("exp_name is required")
    base_cfg = OmegaConf.load("config/base.yaml")
    eval_cfg = OmegaConf.load(f"config/eval_{cli_cfg.get('dataset')}.yaml")
    cfg = OmegaConf.merge(base_cfg, eval_cfg, cli_cfg)
    model_cfg = OmegaConf.load(Path(cfg.exp_root_dir) / "config.yaml")
    cfg = OmegaConf.merge(cfg, model_cfg)
    seed_everything(cfg.seed)
    eval_dir = Path(base_cfg.evaluation_result_root_dir) / cfg.exp_name / get_timestamp()
    eval_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(eval_dir, "eval.log", "INFO")
    logger.info(" ".join([sys.executable] + sys.argv))
    if "note" in cli_cfg:
        logger.info(f"<blue>Note: {cli_cfg.note}</blue>")
    logger.info(f"Evaluation started at {eval_dir}")

    vae = build_vae[cfg.model.vae.name](
        cfg.model.vae,
        Normalizer(cfg.model.vae.mean_fpath, cfg.model.vae.std_fpath),
        # Normalizer(cfg.model.vae.latent_mean_fpath, cfg.model.vae.latent_std_fpath),
        None,
        # Normalizer(cfg.model.vae.latent_mean_fpath, cfg.model.vae.latent_std_fpath),
        checkpoint=(Path(cfg.model.vae.path) / "model" / "latest.pth").as_posix(),
    )
    vae.eval()
    vae.to("cuda")
    logger.info(f"VAE loaded from {cfg.model.vae.path}")

    ckpt = cfg.get("checkpoint", "latest")
    logger.info(f"Loading DIT checkpoint from {Path(cfg.exp_root_dir) / 'model' / f'{ckpt}.pth'}")
    dit = build_dit(
        cfg.model.dit, checkpoint=(Path(cfg.exp_root_dir) / "model" / f"{ckpt}.pth").as_posix()
    )
    dit.eval()
    dit.to("cuda")

    gt_dataset = build_dataset[cfg.eval.dataset_module](
        mode="test", feature_type=cfg.model.vae.feature_type
    )
    if cli_cfg.dataset == "interhuman":
        from model.evaluator import EvaluatorModelWrapper, get_generated_motion_loader
        gt_dataloader = DataLoader(
            gt_dataset, batch_size=cfg.eval.batch_size, num_workers=0, drop_last=True, shuffle=True
        )
        eval_wrapper = EvaluatorModelWrapper(cfg.model.evaluator, device="cuda")
    elif cli_cfg.dataset == "interx":
        from dataset.interx import collate_fn
        from model.evaluator_interx import EvaluatorModelWrapper, get_generated_motion_loader

        gt_dataloader = DataLoader(
            gt_dataset,
            batch_size=cfg.eval.batch_size,
            num_workers=4,
            drop_last=True,
            shuffle=True,
            collate_fn=collate_fn,
        )

        wrapper_cfg = OmegaConf.load("checkpoints/hhi/Comp_v6_KLD01/opt.yaml")
        eval_wrapper = EvaluatorModelWrapper(wrapper_cfg)

    eval_motion_loaders = {}
    eval_motion_loaders["generation"] = lambda: get_generated_motion_loader(
        batch_size=cfg.eval.batch_size,
        vae=vae,
        dit=dit,
        ground_truth_dataset=gt_dataset,
        mm_num_samples=cfg.eval.mm_num_samples,
        mm_num_repeats=cfg.eval.mm_num_repeats,
        cond_scale=cfg.eval.cond_scale,
        feature_type=cfg.model.vae.feature_type,
        history_length=cfg.eval.history_length,
        sampling_timesteps=cfg.eval.sampling_timesteps,
        uncertainty_scale=cfg.eval.uncertainty_scale,
        scheduling_matrix_type=cfg.eval.scheduling_matrix_type,
        infer_from_velocity=cfg.eval.infer_from_velocity,
    )
    evaluation()
