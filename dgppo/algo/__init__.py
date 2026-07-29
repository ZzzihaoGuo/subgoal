from .base import Algorithm
from .informarl import InforMARL
from .informarl_subgoal import InforMARL_SUB
from .informarl_lagr import InforMARLLagr
from .dgppo import DGPPO
from .hcbfcrpo import HCBFCRPO
from .informarl_qmix import InforMARL_QMIX


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
    elif algo == 'informarl_qmix':
        return InforMARL_QMIX(**kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
