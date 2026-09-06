"""
main.py
=======
Step 3 of the DRAO execution sequence - the adversarial loop.

Usage
-----
  python main.py
  python main.py --seed 123
  python main.py --iterations 10

Execution order
---------------
  1. python build_real_embeddings.py
  2. python train_detector.py
  3. python main.py
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import ExperimentConfig, get_config
from src.detector import load_detector
from src.embedding import Wav2VecEmbedder
from src.llm_generator import build_prompt_generator
from src.logging_utils import setup_logger
from src.pipeline import run_pipeline
from src.tts_engine import TTSEngine


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_ready(cfg: ExperimentConfig) -> None:
    logger = logging.getLogger("foeglass.main")
    if not cfg.paths.real_embeddings_path.exists():
        logger.error(
            "[ERROR] Real embeddings not found.\n"
            "  -> Run: python build_real_embeddings.py"
        )
        sys.exit(1)

    if not cfg.paths.detector_path.exists():
        logger.error(
            "[ERROR] Detector checkpoint not found.\n"
            "  -> Run: python train_detector.py"
        )
        sys.exit(1)

    if cfg.llm.provider.lower() == "groq" and not cfg.groq_api_key:
        logger.error(
            "[ERROR] GROQ_API_KEY is not set.\n"
            "  -> Create a .env file in foeglass/ with:\n"
            "     GROQ_API_KEY=gsk_..."
        )
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FoeGlass DRAO - adversarial audio deepfake optimisation"
    )
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (overrides config)")
    p.add_argument("--iterations", type=int, default=None,
                   help="Number of adversarial iterations (overrides config)")
    p.add_argument("--run-name", type=str, default=None,
                   help="Identifier for this run (used in filenames)")
    p.add_argument("--session-root", type=str, default=None,
                   help="Optional session directory that will contain logs/results/plots/audio for this run")
    p.add_argument("--profile", type=str, default="full_drao",
                   choices=["baseline", "no_mmd", "no_smoothness", "full_drao", "custom"],
                   help="Predefined experiment profile")
    p.add_argument("--lambda-mmd", type=float, default=None,
                   help="lambda1 - MMD penalty weight (overrides config)")
    p.add_argument("--lambda-smooth", type=float, default=None,
                   help="lambda2 - smoothness penalty weight (overrides config)")
    p.add_argument("--alpha", type=float, default=None,
                   help="Weight on detector score improvement (overrides config)")
    p.add_argument("--beta", type=float, default=None,
                   help="Weight on absolute detector score (overrides config)")
    p.add_argument("--margin-weight", type=float, default=None,
                   help="Weight on threshold-crossing encouragement term (overrides config)")
    p.add_argument("--attack-detector", type=str, default="A",
                   choices=["A", "B", "default"],
                   help="Detector checkpoint used for optimization")
    p.add_argument("--eval-detector", type=str, default=None,
                   choices=["A", "B", "default"],
                   help="Optional second detector used only for transfer evaluation")
    p.add_argument("--disable-mmd", action="store_true",
                   help="Disable the MMD penalty term")
    p.add_argument("--disable-smoothness", action="store_true",
                   help="Disable the smoothness penalty term")
    p.add_argument("--disable-logit", action="store_true",
                   help="Use detector score instead of detector logit in the objective")
    p.add_argument("--disable-memory", action="store_true",
                   help="Disable elite memory reuse")
    p.add_argument("--disable-margin", action="store_true",
                   help="Disable the margin encouragement term")
    return p.parse_args()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "run"


def _build_run_descriptor(args: argparse.Namespace, seed: int) -> str:
    parts = [
        f"profile-{args.profile}",
        f"seed-{seed}",
        f"attack-{args.attack_detector}",
    ]
    if args.eval_detector is not None:
        parts.append(f"eval-{args.eval_detector}")
    if args.iterations is not None:
        parts.append(f"iters-{args.iterations}")
    return _slugify("_".join(parts))


def _apply_profile(cfg: ExperimentConfig, profile: str) -> None:
    opt = cfg.optimization
    if profile == "baseline":
        opt.use_mmd = False
        opt.use_smoothness = False
        opt.use_logit = False
        opt.use_margin = False
        opt.use_delta_score = False
        opt.use_momentum = False
        opt.normalize_terms = False
        opt.use_memory = True
        opt.memory_metric = "detector_score"
        return
    if profile == "no_mmd":
        opt.use_mmd = False
        opt.use_smoothness = True
        opt.use_logit = False
        opt.use_margin = True
        opt.use_delta_score = True
        opt.use_momentum = True
        opt.normalize_terms = True
        opt.use_memory = True
        opt.memory_metric = "objective"
        return
    if profile == "no_smoothness":
        opt.use_mmd = True
        opt.use_smoothness = False
        opt.use_logit = False
        opt.use_margin = True
        opt.use_delta_score = True
        opt.use_momentum = True
        opt.normalize_terms = True
        opt.use_memory = True
        opt.memory_metric = "objective"
        return
    if profile == "full_drao":
        opt.use_mmd = True
        opt.use_smoothness = True
        opt.use_logit = False
        opt.use_margin = True
        opt.use_delta_score = True
        opt.use_momentum = True
        opt.normalize_terms = True
        opt.use_memory = True
        opt.memory_metric = "objective"


def _resolve_detector_path(cfg: ExperimentConfig, detector_name: str | None) -> Path:
    name = (detector_name or "default").upper()
    if name == "A":
        return cfg.paths.detector_a_path if cfg.paths.detector_a_path.exists() else cfg.paths.detector_path
    if name == "B":
        return cfg.paths.detector_b_path if cfg.paths.detector_b_path.exists() else cfg.paths.detector_path
    return cfg.paths.detector_path


def main() -> None:
    args = _parse_args()
    cfg = get_config()
    _apply_profile(cfg, args.profile)

    if args.seed is not None:
        cfg.optimization.seed = args.seed
    if args.iterations is not None:
        cfg.optimization.iterations = args.iterations
    if args.lambda_mmd is not None:
        cfg.optimization.lambda_mmd = args.lambda_mmd
    if args.lambda_smooth is not None:
        cfg.optimization.lambda_smoothness = args.lambda_smooth
    if args.alpha is not None:
        cfg.optimization.alpha_attack_gain = args.alpha
    if args.beta is not None:
        cfg.optimization.beta_absolute_score = args.beta
    if args.margin_weight is not None:
        cfg.optimization.margin_weight = args.margin_weight
    if args.disable_mmd:
        cfg.optimization.use_mmd = False
    if args.disable_smoothness:
        cfg.optimization.use_smoothness = False
    if args.disable_logit:
        cfg.optimization.use_logit = False
    if args.disable_memory:
        cfg.optimization.use_memory = False
    if args.disable_margin:
        cfg.optimization.use_margin = False

    seed_everything(cfg.optimization.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = args.run_name or _build_run_descriptor(args, cfg.optimization.seed)
    run_name = _slugify(run_label)
    if args.session_root:
        session_root = Path(args.session_root)
    else:
        session_root = cfg.paths.experiments_dir / f"{timestamp}_{run_name}"
    cfg.paths = cfg.paths.for_session(session_root)

    logger = setup_logger("foeglass.main", cfg.paths.logs_dir / f"{run_name}.log")
    _assert_ready(cfg)
    cfg.save_snapshot(cfg.paths.results_dir / run_name / "config.json")
    cfg.save_snapshot(session_root / "session_config.json")
    with (session_root / "session_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": timestamp,
                "session_root": str(session_root),
                "run_name": run_name,
                "profile": args.profile,
                "logs_dir": str(cfg.paths.logs_dir),
                "results_dir": str(cfg.paths.results_dir),
                "plots_dir": str(cfg.paths.plots_dir),
                "audio_dir": str(cfg.paths.audio_dir),
            },
            fh,
            indent=2,
        )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  FoeGlass DRAO  -  {run_name}")
    logger.info(f"  session_root={session_root}")
    logger.info(
        f"  seed={cfg.optimization.seed}  "
        f"profile={args.profile}  "
        f"iters={cfg.optimization.iterations}  "
        f"alpha={cfg.optimization.alpha_attack_gain}  "
        f"beta={cfg.optimization.beta_absolute_score}  "
        f"margin={cfg.optimization.margin_weight}  "
        f"lambda1={cfg.optimization.lambda_mmd}  "
        f"lambda2={cfg.optimization.lambda_smoothness}"
    )
    logger.info(f"{'=' * 60}\n")

    real_embs = np.load(str(cfg.paths.real_embeddings_path)).astype(np.float32)
    logger.info("[main] Real embeddings: %s", real_embs.shape)

    embedder = Wav2VecEmbedder(
        model_name=cfg.model.embedding_model_name,
        sample_rate=cfg.audio.sample_rate,
    )

    detector = load_detector(
        checkpoint=_resolve_detector_path(cfg, args.attack_detector),
        input_dim=real_embs.shape[1],
        hidden_dims=cfg.model.detector_hidden_dims,
        dropout=cfg.model.detector_dropout,
    )
    transfer_detector = None
    if args.eval_detector is not None:
        transfer_detector = load_detector(
            checkpoint=_resolve_detector_path(cfg, args.eval_detector),
            input_dim=real_embs.shape[1],
            hidden_dims=cfg.model.detector_hidden_dims,
            dropout=cfg.model.detector_dropout,
        )

    llm = build_prompt_generator(
        provider=cfg.llm.provider,
        groq_api_key=cfg.groq_api_key,
        groq_model=cfg.llm.groq_model,
        ollama_model=cfg.llm.ollama_model,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        max_tokens=cfg.llm.max_tokens,
        max_chars=cfg.audio.max_prompt_chars,
    )

    tts = TTSEngine(
        model_name=cfg.audio.tts_model_name,
        sample_rate=cfg.audio.sample_rate,
        normalize=cfg.audio.normalize_peak,
    )

    opt = cfg.optimization
    run_pipeline(
        embedder=embedder,
        detector=detector,
        llm=llm,
        tts=tts,
        real_embeddings=real_embs,
        audio_dir=cfg.paths.audio_dir,
        results_dir=cfg.paths.results_dir,
        plots_dir=cfg.paths.plots_dir,
        logs_dir=cfg.paths.logs_dir,
        iterations=opt.iterations,
        alpha_attack_gain=opt.alpha_attack_gain,
        beta_absolute_score=opt.beta_absolute_score,
        margin_weight=opt.margin_weight,
        lambda_mmd=opt.lambda_mmd,
        lambda_smoothness=opt.lambda_smoothness,
        buffer_size=opt.buffer_size,
        memory_top_k=opt.memory_top_k,
        memory_reuse_probability=opt.memory_reuse_probability,
        momentum_gamma=opt.momentum_gamma,
        asr_thresholds=opt.asr_thresholds,
        margin_target=opt.margin_target,
        initial_mmd=opt.initial_mmd_reference,
        run_name=run_name,
        seed=opt.seed,
        use_mmd=opt.use_mmd,
        use_smoothness=opt.use_smoothness,
        use_logit=opt.use_logit,
        use_memory=opt.use_memory,
        use_margin=opt.use_margin,
        use_delta_score=opt.use_delta_score,
        use_momentum=opt.use_momentum,
        normalize_terms=opt.normalize_terms,
        memory_metric=opt.memory_metric,
        profile_name=args.profile,
        transfer_detector=transfer_detector,
        config_snapshot=cfg.to_dict(),
    )


if __name__ == "__main__":
    main()
