#!/usr/bin/env python3

"""
differentially privately train a model
"""

import pathlib

import opacus
from transformers import EarlyStoppingCallback, TrainingArguments

from cotorra.trainer import Trainer, TrainerWithCustomLoss


class TrainerWithCustomLossDP(TrainerWithCustomLoss):
    """Opacus-compatible trainer that can use a DP-wrapped train DataLoader."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dp_train_dataloader = None

    def set_dp_train_dataloader(self, dataloader):
        self._dp_train_dataloader = dataloader

    def get_train_dataloader(self):
        if self._dp_train_dataloader is not None:
            return self._dp_train_dataloader
        return super().get_train_dataloader()

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.args.gradient_accumulation_steps != 1:
            raise ValueError(
                "Opacus DP training currently requires gradient_accumulation_steps=1."
            )

        self.optimizer.zero_grad()
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        self.accelerator.backward(loss)
        return loss.detach()


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

        self.trainer = TrainerWithCustomLossDP(
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

        # HuggingFace Trainer calls get_train_dataloader() internally, so we must
        # ensure get_train_dataloader() returns the Opacus-wrapped DataLoader.
        privacy_engine = opacus.PrivacyEngine()
        dp_model, dp_optimizer, dp_dataloader = privacy_engine.make_private(
            module=self.trainer.model,
            optimizer=self.trainer.optimizer,
            data_loader=self.trainer.get_train_dataloader(),
            noise_multiplier=self.cfg.get("privacy_parameters", {}).get(
                "noise_multiplier", 1.0
            ),
            max_grad_norm=self.cfg.get("privacy_parameters", {}).get(
                "max_grad_norm", 1.0
            ),
            poisson_sampling=False,
        )
        self.trainer.model = dp_model
        self.trainer.optimizer = dp_optimizer
        self.trainer.set_dp_train_dataloader(dp_dataloader)
        self.model = self.trainer.model
        self.privacy_engine = privacy_engine


if __name__ == "__main__":
    self = TrainerDP(
        processed_data_home="./processed/mimic", output_home="./output/mimic"
    )
    # self.train(verbose=True)
    breakpoint()
