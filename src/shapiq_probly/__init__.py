"""shapiq-probly: explain which features make a model uncertain, and why.

Uncertainty measures become value functions of cooperative games over features;
Shapley values and interactions of the total, aleatoric and epistemic games
tell whether a feature contributes noise or missing knowledge.
"""

from .adapters import as_member_predictor, from_members, from_sklearn
from .exact import all_coalitions, compute_index, moebius_transform
from .explainer import UncertaintyExplainer, explain_uncertainty
from .explanation import Attribution, UncertaintyExplanation
from .games import UncertaintyGame
from .imputation import GaussianSampler, KNNSampler, MarginalSampler, Sampler
from .measures import entropy_decomposition
from .plots import plot_interaction_network, plot_uncertainty_explanation, plot_uncertainty_profile
from .synthetic import make_uncertainty_benchmark

__version__ = "0.1.0"

__all__ = [
    "Attribution",
    "GaussianSampler",
    "KNNSampler",
    "MarginalSampler",
    "Sampler",
    "UncertaintyExplainer",
    "UncertaintyExplanation",
    "UncertaintyGame",
    "all_coalitions",
    "as_member_predictor",
    "compute_index",
    "entropy_decomposition",
    "explain_uncertainty",
    "from_members",
    "from_sklearn",
    "make_uncertainty_benchmark",
    "moebius_transform",
    "plot_interaction_network",
    "plot_uncertainty_explanation",
    "plot_uncertainty_profile",
]
