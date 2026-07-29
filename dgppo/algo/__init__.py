from .base import Algorithm
from .informarl import InforMARL
from .informarl_subgoal import InforMARL_SUB
from .informarl_lagr import InforMARLLagr
from .dgppo import DGPPO
from .hcbfcrpo import HCBFCRPO
from .informarl_matd3 import InforMARL_MATD3


def make_algo(algo: str, **kwargs) -> Algorithm:
    if algo == 'informarl':
        return InforMARL(**kwargs)
    elif algo == 'informarl_subgoal':
        return InforMARL_SUB(**kwargs)
    elif algo == 'informarl_lagr':
        return InforMARLLagr(**kwargs)
    elif algo == 'dgppo':
        return DGPPO(**kwargs)
    elif algo == 'hcbfcrpo':
        return HCBFCRPO(**kwargs)
    elif algo == 'informarl_matd3':
        return InforMARL_MATD3(**kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
