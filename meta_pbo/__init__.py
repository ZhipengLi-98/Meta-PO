"""Meta-PBO: Transfer Learning for Preference-Based Bayesian Optimization"""

__version__ = '1.0.0'

from .optimizers import SequentialPBO, TransferPBO
from .transfer import TAF_M, TAF_R

__all__ = ['SequentialPBO', 'TransferPBO', 'TAF_M', 'TAF_R']
