#!/usr/bin/env python3
"""Promote a registered model to Staging or Production with gate checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

from mlops.model_registry import ModelRegistry

logger = structlog.get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument override.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Promote a NEXUS-CV registered model.")
    parser.add_argument("--model-name", required=True, help="MLflow registered model name")
    parser.add_argument("--version", required=True, help="Model version to promote")
    parser.add_argument(
        "--to-stage",
        choices=["staging", "production"],
        required=True,
        help="Target registry stage",
    )
    parser.add_argument("--force", action="store_true", help="Skip promotion gate checks")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate gates and print decision without applying changes",
    )
    parser.add_argument("--map50", type=float, default=0.0, help="Detection mAP50 metric")
    parser.add_argument("--ade-m", type=float, default=999.0, help="LSTM ADE metric in meters")
    parser.add_argument(
        "--comment",
        default="promoted via promote_model.py",
        help="Staging comment",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argument override.

    Returns:
        Exit code (1 when promotion blocked).
    """
    args = _parse_args(argv)
    registry = ModelRegistry()
    metrics = {"mAP50": args.map50, "ade_m": args.ade_m}

    if args.to_stage == "production":
        can_promote, reason = registry.evaluate_promotion(
            args.model_name,
            args.version,
            metrics,
            force=args.force,
        )
        print(f"Promotion decision: {'ALLOW' if can_promote else 'BLOCK'}")
        print(f"Reason: {reason}")

        if args.dry_run:
            return 0 if can_promote or args.force else 1

        if not can_promote and not args.force:
            return 1

        registry.promote_to_production(
            args.model_name,
            args.version,
            metrics,
            force=args.force,
        )
        print(f"Promoted {args.model_name} v{args.version} to Production")
        return 0

    if args.dry_run:
        print(f"Dry-run: would transition {args.model_name} v{args.version} to Staging")
        print(f"Comment: {args.comment}")
        return 0

    registry.transition_to_staging(args.model_name, args.version, comment=args.comment)
    print(f"Transitioned {args.model_name} v{args.version} to Staging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
