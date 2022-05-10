#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, List, Dict, Tuple, Any, Optional
from torch.cuda.amp import autocast

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments, )
from transformers.trainer_pt_utils import LabelSmoother

from transformers.trainer import *

from uie.seq2seq.constraint_decoder import get_constraint_decoder


@dataclass
class ConstraintSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Parameters:
        constraint_decoding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use Constraint Decoding
        structure_weight (:obj:`float`, `optional`, defaults to :obj:`None`):
    """
    constraint_decoding: bool = field(default=False, metadata={"help": "Whether to Constraint Decoding or not."})
    save_better_checkpoint: bool = field(default=False,
                                         metadata={"help": "Whether to save better metric checkpoint"})
    start_eval_step: int = field(default=0, metadata={"help": "Start Evaluation after Eval Step"})


class ConstraintSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, decoding_type_schema=None, task='event', decoding_format='tree', source_prefix=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoding_format = decoding_format
        self.decoding_type_schema = decoding_type_schema

        # Label smoothing by sum token loss, different from different Label smootheing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
            print('Using %s' % self.label_smoother)
        else:
            self.label_smoother = None

        if self.args.constraint_decoding:
            self.constraint_decoder = get_constraint_decoder(tokenizer=self.tokenizer,
                                                             type_schema=self.decoding_type_schema,
                                                             decoding_schema=self.decoding_format,
                                                             source_prefix=source_prefix,
                                                             task_name=task)
        else:
            self.constraint_decoder = None

        self.oom_batch = 0

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        oom = False
        oom_message = ""
        try:
            loss = super().training_step(model, inputs)
            return loss
        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
                oom_message = str(e)
                logger.warning(f'ran out of memory {self.oom_batch} on {self.args.local_rank}')
                for k, v in inputs.items():
                    print(k, v.size())
            else:
                raise e

        if oom:
            self.oom_batch += 1
            raise RuntimeError(oom_message)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            **kwargs
        )

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        if self.args.start_eval_step > 0 and self.state.global_step < self.args.start_eval_step:
            return

        previous_best_metric = self.state.best_metric
        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        # Only save the checkpoint better than previous_best_metric
        if self.args.save_better_checkpoint and self.args.metric_for_best_model is not None:
            if metrics is not None and previous_best_metric is not None:
                if metrics[self.args.metric_for_best_model] <= previous_best_metric:
                    return

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        def prefix_allowed_tokens_fn(batch_id, sent):
            # print(self.tokenizer.convert_ids_to_tokens(inputs['labels'][batch_id]))
            src_sentence = inputs['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss, generated_tokens, labels


def main(): pass


if __name__ == "__main__":
    main()
