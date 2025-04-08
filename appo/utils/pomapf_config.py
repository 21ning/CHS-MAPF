try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pogema import GridConfig


class POMAPFConfig(GridConfig):
    integration: Literal['SampleFactory'] = 'SampleFactory'
    collision_system: Literal['priority'] = 'priority'
    observation_type: Literal['POMAPF', 'MAPF'] = 'POMAPF'
