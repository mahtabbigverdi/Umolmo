import dataclasses

from olmo.he_molmo.he_molmo import HeMolmoConfig
from olmo.train.trainer_config import _TrainerConfig


@dataclasses.dataclass
class HeMolmoTrainerConfig(_TrainerConfig):
    model: HeMolmoConfig = dataclasses.field(default_factory=HeMolmoConfig)
    name: str = "he_molmo"
