"""
Subclass trainer that adds a Sudoku rule-based auxiliary loss.

Reimplements inner_forward (duplicating the ~30-line body of the parent)
rather than modifying the parent's return signature — keeps the baseline
code path in trainer.py completely untouched.

Total loss per step:
    loss = ce_loss_weighted + rule_loss_weight * rule_loss

Components are stashed and injected into the HF Trainer's log dict via an
override of `log()` so they appear in train.log / trainer_state.json.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from llmtuner.tuner.mdm.trainer import CustomDiffusionTrainer
from llmtuner.tuner.mdm.rule_loss import compute_rule_loss, digit_token_ids


class RuleLossDiffusionTrainer(CustomDiffusionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_ce_loss: float = 0.0
        self._last_rule_loss: float = 0.0
        self._last_lambda: float = float(self.diff_args.rule_loss_weight)
        self._digit_ids_cached: torch.Tensor = None

    def _digit_ids(self, device: torch.device) -> torch.Tensor:
        if self._digit_ids_cached is None or self._digit_ids_cached.device != device:
            self._digit_ids_cached = torch.tensor(
                digit_token_ids(self.tokenizer), device=device, dtype=torch.long
            )
        return self._digit_ids_cached

    def inner_forward(self, model, inputs):
        """Duplicate of CustomDiffusionTrainer.inner_forward + rule loss term."""
        x = inputs["input_ids"]
        src_mask = inputs["src_mask"].bool()
        batch_size = x.size(0)

        if isinstance(model, DDP):
            vocab_size = model.module.vocab_size
        else:
            vocab_size = model.vocab_size
        num_timesteps = self.diff_args.diffusion_steps

        t = torch.randint(0, num_timesteps, (batch_size,), device=x.device)
        x_t, t, loss_mask = self.q_sample(inputs, t, maskable_mask=~src_mask)

        attention_mask = torch.ones_like(x_t)

        logits = model(x_t, t, attention_mask=attention_mask)
        logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)

        ce = F.cross_entropy(
            logits.reshape(-1, vocab_size), x.reshape(-1), reduction="none"
        ).float()
        ce = ce.masked_fill(~loss_mask.reshape(-1), 0)

        if self.diff_args.token_reweighting:
            ce = self.diff_args.alpha * (1 - torch.exp(-ce)) ** self.diff_args.gamma * ce

        if self.diff_args.time_reweighting == 'original':
            weight = 1 / (t + 1)[:, None].float()
        elif self.diff_args.time_reweighting == 'linear':
            weight = (num_timesteps - t)[:, None].float()
        else:
            weight = t.new_ones((batch_size, 1)).float()

        weight = weight.expand(loss_mask.size())
        ce_loss = (ce * weight.reshape(-1)).sum() / loss_mask.sum()

        # --- rule loss (the only addition vs. the parent) ---
        digit_ids = self._digit_ids(logits.device)
        rule_loss = compute_rule_loss(
            logits, src_mask=inputs["src_mask"], tokenizer=self.tokenizer, digit_ids=digit_ids
        )

        lam_base = float(self.diff_args.rule_loss_weight)
        schedule = getattr(self.diff_args, "rule_loss_schedule", "constant")
        if schedule == "linear":
            max_steps = max(1, int(self.args.max_steps))
            frac = min(1.0, self.state.global_step / max_steps)
            lam = lam_base * max(0.0, 1.0 - frac)
        else:
            lam = lam_base

        total = ce_loss + lam * rule_loss

        # Stash for log() override
        self._last_ce_loss = ce_loss.detach().float().item()
        self._last_rule_loss = rule_loss.detach().float().item()
        self._last_lambda = lam

        return total

    def log(self, logs, *args, **kwargs):
        """Inject per-step ce_loss / rule_loss snapshots into HF Trainer's log dict."""
        logs = dict(logs)
        logs["ce_loss"] = self._last_ce_loss
        logs["rule_loss"] = self._last_rule_loss
        logs["rule_loss_weight"] = self._last_lambda  # effective λ (after schedule)
        logs["rule_loss_weight_base"] = float(self.diff_args.rule_loss_weight)
        return super().log(logs, *args, **kwargs)
