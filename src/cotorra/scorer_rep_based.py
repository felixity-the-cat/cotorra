#!/usr/bin/env python3

"""
make representation-based predictions on held-out data
"""

import pathlib
import typing

import lightgbm as lgb
import numpy as np
import polars as pl
import sklearn as skl
import tqdm
import xgboost as xgb
from omegaconf import OmegaConf

from cotorra.configurable import Configurable


class RepBasedScorer(Configurable):
    default_file = "scoring.yaml"

    def __init__(
        self,
        scoring_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        model_home: pathlib.Path | str = None,
        output_home: pathlib.Path | str = None,
        estimator_type: typing.Literal[
            "k-NN", "lightGBM", "logistic", "logistic-CV", "XGBoost"
        ] = "lightGBM",
        **kwargs,
    ):
        super().__init__(scoring_cfg, **kwargs)
        self.processed_data_home, self.model_home = map(
            lambda x: pathlib.Path(x).expanduser().resolve(),
            (processed_data_home, model_home),
        )
        self.output_home = (
            pathlib.Path(output_home).expanduser().resolve()
            if output_home is not None
            else self.processed_data_home
        ) / f"scores-rep-based-{self.model_home.name}.parquet"
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")

        self.splits = ("train", "tuning", "held_out")
        self.estimator_type = estimator_type

        try:
            self.features = {
                s: np.vstack(
                    pl.scan_parquet(
                        self.processed_data_home
                        / f"features-{s}*-{self.model_home.name}.parquet"
                    )
                    .select("features")
                    .collect()
                    .to_series()
                    .to_list()
                )
                for s in self.splits
            }
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Expected extracted features at: "
                f"{self.processed_data_home / 'features-<split>-<model_name>.parquet'},"
                " but not found."
                " Please run `cotorra extract` first."
            ) from e

        self.labels = {
            s: pl.scan_parquet(self.processed_data_home / f"{s}_for_inference.parquet")
            for s in self.splits
        }

    def score_label(self, target_token="DSCG//expired"):
        cols = (~pl.col(f"{target_token}_past"), f"{target_token}_future")
        train_valid, train_label = (
            self.labels["train"].select(*cols).collect().to_numpy().T
        )
        tuning_valid, tuning_label = (
            self.labels["tuning"].select(*cols).collect().to_numpy().T
        )
        held_out_valid = (
            self.labels["held_out"].select(cols[0]).collect().to_numpy().ravel()
        )

        match str(self.estimator_type).lower():
            case "logistic" | "lr" | "logistic-regression":
                self.logger.info("Using logistic regression classifier")
                mdl = skl.pipeline.make_pipeline(
                    skl.preprocessing.StandardScaler(),
                    skl.linear_model.LogisticRegression(max_iter=10_000),
                )
            case "logistic-cv" | "lr-cv":
                self.logger.info(
                    "Using logistic regression classifier with cross-validation"
                )
                mdl = skl.pipeline.make_pipeline(
                    skl.preprocessing.StandardScaler(),
                    skl.linear_model.LogisticRegressionCV(n_jobs=-1, max_iter=10_000),
                )
            case "k-nn" | "knn" | "k_nn":
                self.logger.info("Using k-nn classifier")
                mdl = skl.neighbors.KNeighborsClassifier(
                    n_neighbors=max(25, int(0.2 * sum(train_valid))), n_jobs=-1
                )
            case "xgboost":
                self.logger.info("Using XGBoost classifier")
                mdl = xgb.XGBClassifier(
                    min_child_weight=5, max_leaves=64, n_estimators=250, n_jobs=-1
                )
            case _:
                self.logger.info("Using (default) lightGBM classifier")
                mdl = lgb.LGBMClassifier(
                    min_data_in_leaf=5, num_leaves=64, n_estimators=250, n_jobs=-1
                )

        mdl.fit(
            X=self.features["train"][train_valid],
            y=train_label[train_valid],
            **(
                {
                    "eval_set": [
                        (
                            self.features["tuning"][tuning_valid],
                            tuning_label[tuning_valid],
                        )
                    ],
                    "eval_metric": "auc",
                }
                if str(self.estimator_type).lower() in ("lightgbm", "xgboost")
                else {}
            ),
        )

        scores = np.nan * np.ones_like(held_out_valid)
        scores[held_out_valid] = mdl.predict_proba(
            X=self.features["held_out"][held_out_valid]
        )[:, 1]

        return scores

    def score(self):
        res = dict()
        for tt in tqdm.tqdm(self.cfg.score.target_tokens, position=0):
            res[f"{tt}_rep_score"] = self.score_label(target_token=tt)

        return res

    def save_all(self, verbose: bool = False):
        (
            df_res := self.labels["held_out"].with_columns(pl.from_dict(self.score()))
        ).sink_parquet(self.output_home)

        if verbose:
            self.logger.summarize_preds(df_res, self.cfg.score.target_tokens)


if __name__ == "__main__":
    self = RepBasedScorer()
    self.save_all(verbose=True)
    # breakpoint()
