"""Generate a two-person interaction followed by another interaction."""

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from dataset.utils import Normalizer, features_to_interhuman
from model.dit import build_dit
from model.vae import build_vae
from utils import seed_everything


def main():
    cli_cfg = OmegaConf.from_cli()
    exp_name = cli_cfg.get("exp_name", "dit_without-shape")

    base_cfg = OmegaConf.load("config/base.yaml")
    model_cfg = OmegaConf.load(Path(base_cfg.checkpoint_root_dir) / exp_name / "config.yaml")
    cfg = OmegaConf.merge(base_cfg, model_cfg, cli_cfg)

    device = str(cli_cfg.get("device", "cuda"))
    first_prompt = str(
        cli_cfg.get("first_prompt", "one pivots and lifts their right leg to strike")
    )
    second_prompt = str(
        cli_cfg.get(
            "second_prompt",
            "they show respect to each other by standing still and bowing their heads.",
        )
    )
    first_length = int(cli_cfg.get("first_length", 18))
    second_length = int(cli_cfg.get("second_length", 20))
    history_length = int(cli_cfg.get("history_length", 1))
    cond_scale = float(cli_cfg.get("cond_scale", 2.5))
    sampling_timesteps = int(cli_cfg.get("sampling_timesteps", 50))
    uncertainty_scale = float(cli_cfg.get("uncertainty_scale", 5.0))
    scheduling_matrix_type = str(cli_cfg.get("scheduling_matrix_type", "pyramid"))
    output_dir = Path(cli_cfg.get("output_dir", Path(cfg.demo_dir) / "double_to_double"))
    export_video = bool(cli_cfg.get("export_video", False))
    video_path = Path(cli_cfg.get("video_path", output_dir / "demo.mp4"))
    fps = int(cli_cfg.get("fps", cfg.target_framerate))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg.seed))
    vae = build_vae(
        cfg.model.vae,
        normalizer=Normalizer(cfg.model.vae.mean_fpath, cfg.model.vae.std_fpath),
        latent_normalizer=None,
        checkpoint=(Path(cfg.model.vae.path) / "model" / "latest.pth").as_posix(),
    ).eval().to(device)
    dit = build_dit(
        cfg.model.dit,
        checkpoint=Path(cfg.exp_root_dir) / "model" / "latest.pth",
    ).eval().to(device)

    with torch.no_grad():
        latent1, latent2 = dit.generate(
            [first_prompt],
            torch.tensor([[first_length, first_length]]),
            cond_scale=cond_scale,
            scheduling_matrix_type=scheduling_matrix_type,
            sampling_timesteps=sampling_timesteps,
            uncertainty_scale=uncertainty_scale,
            n_persons=2,
        )
        latent1, latent2 = dit.generate(
            [second_prompt],
            torch.tensor([[second_length, second_length]]),
            cond_scale=cond_scale,
            scheduling_matrix_type=scheduling_matrix_type,
            sampling_timesteps=sampling_timesteps,
            uncertainty_scale=uncertainty_scale,
            history_length=history_length,
            context1=latent1,
            context2=latent2,
            n_persons=2,
        )
        features1 = vae.decode(latent1, semantic_idx=0).cpu().squeeze(0)
        features2 = vae.decode(latent2, semantic_idx=0).cpu().squeeze(0)

    motion1, motion2 = features_to_interhuman(
        features1,
        features2,
        feature_type=cfg.model.vae.feature_type,
    )
    motion1, motion2 = motion1.cpu().numpy(), motion2.cpu().numpy()
    joints1 = motion1[:, : 22 * 3].reshape(-1, 22, 3)
    joints2 = motion2[:, : 22 * 3].reshape(-1, 22, 3)
    transition_frame = first_length * int(cfg.model.vae.stride_t) ** int(cfg.model.vae.down_t)

    np.save(output_dir / "motion.npy", np.stack([motion1, motion2]))
    np.save(output_dir / "joints.npy", np.stack([joints1, joints2], axis=1))
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "mode": "double_to_double",
                "prompts": [first_prompt, second_prompt],
                "token_lengths": [first_length, second_length],
                "transition_frame": transition_frame,
                "seed": int(cfg.seed),
            },
            indent=2,
        )
        + "\n"
    )
    if export_video:
        from utils.body_model import BodyModel
        from utils.visualizer import Visualizer

        video_path.parent.mkdir(parents=True, exist_ok=True)
        body_model = BodyModel(bm_fname=cfg.smplh_model_fpath)
        text = [first_prompt] * transition_frame + [second_prompt] * (
            len(joints1) - transition_frame
        )
        Visualizer.export_skeleton(
            np.stack([joints1, joints2]),
            kinetree=body_model.kintree_table[:22],
            fname=video_path.as_posix(),
            fps=fps,
            text=text,
        )
        print(f"Saved visualization to {video_path}")
    print(f"Saved demo to {output_dir}")


if __name__ == "__main__":
    main()
