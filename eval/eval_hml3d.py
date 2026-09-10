import torch
from tqdm import tqdm
from dataset.utils import features_to_interhuman
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from utils import get_logger, get_timestamp, seed_everything
from model import build_vae
from model.dit import build_dit
from dataset.utils import Normalizer
from dataset.dataset_eval_t2m import DATALoader, interhuman_to_hml3d272
import warnings
from scipy import linalg


# Single-GPU evaluation of text to motion model (test time)：
@torch.no_grad()
def evaluation_single(val_loader, vae, dit, evaluator):
    textencoder, motionencoder = evaluator
    dit.eval()
    vae.eval()
    dit.to("cuda")
    vae.to("cuda")
    device = "cuda"

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = torch.tensor([0, 0, 0], device=device)
    R_precision = torch.tensor([0, 0, 0], device=device)
    matching_score_real = torch.tensor(0.0, device=device)
    matching_score_pred = torch.tensor(0.0, device=device)

    nb_sample = torch.tensor(0, device=device)

    for batch in tqdm(val_loader, total=len(val_loader), desc="Evaluating"):
        text, pose, m_length = batch
        bs, seq = pose.shape[:2]
        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(device)
        pred_len = torch.ones(bs).long()

        for k in range(bs):
            latent, _ = dit.generate(
                text[k : k + 1],
                torch.tensor([[m_length[k] // 4, 0]]),
                2.5,
                sampling_timesteps=50,
                uncertainty_scale=5.0,
                scheduling_matrix_type="pyramid",
                n_persons=1
            )

            with torch.no_grad():
                output = vae.decode(latent, semantic_idx=0).detach().cpu()

            output_list = []
            for o in output:
                o_interhuman = features_to_interhuman(
                    o,
                    feature_type="velocity_relative_translation",
                    inference_from_velocity=False,
                    local_dim=262
                )
                output_list.append(torch.from_numpy(val_loader.dataset.forward_transform(interhuman_to_hml3d272(o_interhuman.numpy()))))
            motion_output = torch.stack(output_list)
            pred_len[k] = min(motion_output.shape[1], seq)
            pred_pose_eval[k : k + 1, :motion_output.shape[1]] = motion_output[:, :seq]

        et_pred, em_pred = textencoder(text).loc, motionencoder(pred_pose_eval, pred_len).loc

        pose = pose.to(device).float()
        et, em = textencoder(text).loc, motionencoder(pose, m_length).loc
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R, temp_match = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision_real += torch.tensor(temp_R, device=device)
        matching_score_real += torch.tensor(temp_match, device=device)
        temp_R, temp_match = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision += torch.tensor(temp_R, device=device)
        matching_score_pred += torch.tensor(temp_match, device=device)
        nb_sample += et.shape[0]

        pose = torch.tensor(pose).to(device)

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eval. :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity Pred. {diversity:.4f}, R_precision Real. {R_precision_real}, R_precision Pred. {R_precision}, MM-dist (matching_score) Real. {matching_score_real}, MM-dist (matching_score) Pred. {matching_score_pred}"
    logger.info(msg)

    return (
        fid,
        diversity,
        R_precision[0],
        R_precision[1],
        R_precision[2],
        matching_score_pred,
        logger,
    )


def euclidean_distance_matrix(matrix1, matrix2):
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    dists = np.sqrt(d1 + d2 + d3)
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0),
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0),
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist


if __name__ == "__main__":
    import sys

    sys.path.append("/home/kksix/workspaces/lyc/model")
    sys.path.append("/home/kksix/workspaces/lyc/model/mld")
    warnings.filterwarnings("ignore")

    val_loader = DATALoader(True, 32)

    from model.mld.models.architectures.temos.textencoder.distillbert_actor import (
        DistilbertActorAgnosticEncoder,
    )
    from model.mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

    modelpath = "distilbert-base-uncased"

    textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
    motionencoder = ActorAgnosticEncoder(
        nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300
    )

    ckpt_path = "checkpoints/epoch=99.ckpt"
    print(f"Loading evaluator checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    # load textencoder
    textencoder_ckpt = {}
    for k, v in ckpt["state_dict"].items():
        if k.split(".")[0] == "textencoder":
            name = k.replace("textencoder.", "")
            textencoder_ckpt[name] = v
    textencoder.load_state_dict(textencoder_ckpt, strict=True)
    textencoder.eval()
    textencoder.to("cuda")

    # load motionencoder
    motionencoder_ckpt = {}
    for k, v in ckpt["state_dict"].items():
        if k.split(".")[0] == "motionencoder":
            name = k.replace("motionencoder.", "")
            motionencoder_ckpt[name] = v
    motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
    motionencoder.eval()
    motionencoder.to("cuda")
    # --------------------------------

    evaluator = [textencoder, motionencoder]

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []

    cli_cfg = OmegaConf.from_cli()
    if "exp_name" not in cli_cfg:
        raise ValueError("exp_name is required")
    base_cfg = OmegaConf.load("config/base.yaml")
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
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
        None,
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

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger = evaluation_single(
        val_loader, vae, dit, evaluator
    )
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)

    logger.info("final result:")
    logger.info(f"fid: {fid}")
    logger.info(f"div: {div}")
    logger.info(f"top1: {top1}")
    logger.info(f"top2: {top2}")
    logger.info(f"top3: {top3}")
    logger.info(f"MM-dist (matching score) : {matching}")
