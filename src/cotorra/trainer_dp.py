#!/usr/bin/env python3

"""
differentially privately train a model
"""

import pathlib

import opacus
from transformers import EarlyStoppingCallback, TrainingArguments

from cotorra.trainer import Trainer, TrainerWithCustomLoss


class TrainerDP(Trainer):
    def __init__(
        self,
        training_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        output_home: pathlib.Path | str = None,
        **kwargs,
    ):
        super().__init__(
            training_cfg=training_cfg,
            processed_data_home=processed_data_home,
            output_home=output_home,
            **kwargs,
        )

        self.model = self.model_init()

        self.trainer = TrainerWithCustomLoss(
            model=self.model,
            data_collator=self.collate_fn,
            compute_loss_func=self.loss,
            train_dataset=self.loader.get_train_data(),
            eval_dataset=self.loader.get_tuning_data(),
            args=TrainingArguments(
                output_dir=str(self.output_home), **self.cfg.training_args
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        self.trainer.create_optimizer()
        self.privacy_engine = opacus.PrivacyEngine()

        self.trainer.model, self.trainer.optimizer, self.trainer.train_dataloader = (
            self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.trainer.optimizer,
                data_loader=self.trainer.get_train_dataloader(),
                noise_multiplier=1.0,
                max_grad_norm=1.0,
            )
        )


if __name__ == "__main__":
    self = TrainerDP(
        processed_data_home="./processed/mimic", output_home="./output/mimic"
    )
    self.train(verbose=True)
    # breakpoint()
