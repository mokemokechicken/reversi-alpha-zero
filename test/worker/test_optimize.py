from nose.tools import eq_

from reversi_zero.config import Config
from reversi_zero.worker.optimize import OptimizeWorker


def test_decide_learning_rate():
    config = Config()
    optimizer = OptimizeWorker(config)

    config.trainer.lr_schedules = [
            (0, 0.02),
            (100000, 0.002),
            (200000, 0.0002),
    ]

    eq_(0.02, optimizer.decide_learning_rate(100))
    eq_(0.02, optimizer.decide_learning_rate(99999))
    eq_(0.002, optimizer.decide_learning_rate(100001))
    eq_(0.002, optimizer.decide_learning_rate(199999))
    eq_(0.0002, optimizer.decide_learning_rate(200001))
