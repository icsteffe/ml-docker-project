"""
GLUE Transformer model using PyTorch Lightning.
"""
from datetime import datetime
from typing import Optional

import evaluate
import lightning as L
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


class GLUETransformer(L.LightningModule):
    """PyTorch Lightning module for GLUE tasks."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        optimizer_type: str = "adamw_torch",
        lr_scheduler_type: str = "linear",
        classifier_dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)

        # Set hidden dropout if provided (matches Project 1 notebook)
        if classifier_dropout is not None:
            self.config.hidden_dropout_prob = classifier_dropout

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(self.validation_step_outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            self.validation_step_outputs.clear()
            return loss

        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Use optimizer_type and adam betas from config
        optimizer_type = self.hparams.optimizer_type if hasattr(self.hparams, 'optimizer_type') else "adamw_torch"
        adam_beta1 = self.hparams.adam_beta1 if hasattr(self.hparams, 'adam_beta1') else 0.9
        adam_beta2 = self.hparams.adam_beta2 if hasattr(self.hparams, 'adam_beta2') else 0.999

        if optimizer_type == "adamw_torch":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                betas=(adam_beta1, adam_beta2)
            )
        else:
            # Default to AdamW if optimizer type not recognized
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                betas=(adam_beta1, adam_beta2)
            )

        # Calculate warmup steps from ratio if provided
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_steps
        if hasattr(self.hparams, 'warmup_ratio') and self.hparams.warmup_ratio > 0:
            warmup_steps = int(self.hparams.warmup_ratio * total_steps)

        # Use lr_scheduler_type from config
        lr_scheduler_type = self.hparams.lr_scheduler_type if hasattr(self.hparams, 'lr_scheduler_type') else "linear"

        if lr_scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            # Default to linear if scheduler type not recognized
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]