import dataclasses

from olmo.train.trainer import BeakerLogger


@dataclasses.dataclass
class MockExperiment:
    description: str

    @staticmethod
    def set_description(ex, description: str):
        ex.description = description


@dataclasses.dataclass
class MockBeaker:
    experiment: MockExperiment


def test_beaker_logger():
    mock_url1 = "https://wandb.ai/prior-ai2/cockatoo/runs/xaj"
    mock_url2 = "https://wandb.ai/prior-ai2/cockatoo/runs/39f"
    ex = MockExperiment("Original")
    logger = BeakerLogger(MockBeaker(ex), ex, 10, ex.description)
    logger.log_init()
    assert ex.description == "[Init] Original"
    logger.add_wandb(mock_url1)
    assert ex.description == f"[Init] Original ({mock_url1})"

    logger.add_wandb(mock_url2)
    assert ex.description == f"[Init] Original ({mock_url2})"
