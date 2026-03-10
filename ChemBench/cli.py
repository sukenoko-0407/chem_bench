from __future__ import annotations

import argparse
import json

from .api import fit, predict


def _split_algorithms(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chembench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--train-csv", required=True)
    fit_parser.add_argument("--output-dir", required=True)
    fit_parser.add_argument("--smiles-col", required=True)
    fit_parser.add_argument("--label-col", required=True)
    fit_parser.add_argument("--feature-set", default="ecfp4_2048")
    fit_parser.add_argument("--algorithms", default=None)
    fit_parser.add_argument("--config-path", default=None)
    fit_parser.add_argument("--tuning", action="store_true")
    fit_parser.add_argument("--tuning-config-path", default=None)

    pred_parser = subparsers.add_parser("predict")
    pred_parser.add_argument("--input-csv", required=True)
    pred_parser.add_argument("--model-dir", required=True)
    pred_parser.add_argument("--smiles-col", required=True)
    pred_parser.add_argument("--output-csv", default=None)
    pred_parser.add_argument("--algorithms", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "fit":
        result = fit(
            train_csv=args.train_csv,
            output_dir=args.output_dir,
            smiles_col=args.smiles_col,
            label_col=args.label_col,
            feature_set=args.feature_set,
            algorithms=_split_algorithms(args.algorithms),
            config_path=args.config_path,
            tuning=args.tuning,
            tuning_config_path=args.tuning_config_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    result = predict(
        input_csv=args.input_csv,
        model_dir=args.model_dir,
        smiles_col=args.smiles_col,
        output_csv=args.output_csv,
        algorithms=_split_algorithms(args.algorithms),
    )
    print(result.head())


if __name__ == "__main__":
    main()

