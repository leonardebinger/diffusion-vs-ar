from dataclasses import dataclass, field

@dataclass
class DiffusionArguments:
    r"""
    Arguments of Diffusion Models.
    """
    diffusion_steps: int = field(
        default=64,
        metadata={"help": "timesteps of diffusion models."}
    )
    decoding_strategy: str = field(
        default="stochastic0.5-linear",
        metadata={"help": "<topk_mode>-<schedule>"}
    )
    token_reweighting: bool = field(
        default=False,
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    alpha: float = field(
        default=0.25,
        metadata={"help": "for focal loss"}
    )
    gamma: float = field(
        default=2,
        metadata={"help": "for focal loss"}
    )
    time_reweighting: str = field(
        default='original',
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    topk_decoding: bool = field(
        default=False,
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    rule_loss_weight: float = field(
        default=0.0,
        metadata={"help": "λ for the Sudoku rule-based auxiliary loss. "
                          "0 (default) = baseline (rule loss off)."}
    )
    rule_loss_schedule: str = field(
        default="constant",
        metadata={"help": "Schedule for λ over training. "
                          "'constant' (default) = fixed at rule_loss_weight. "
                          "'linear' = decays linearly from rule_loss_weight to 0 over max_steps."}
    )
    rule_loss_kind: str = field(
        default="collision",
        metadata={"help": "Which rule-based auxiliary loss to compute. "
                          "'collision' (default) = per-pair expected digit collision in [0, 1]. "
                          "'permanent' = -log(perm(M)) summed over 27 peer sets (Ryser formula)."}
    )

    def __post_init__(self):
        pass
