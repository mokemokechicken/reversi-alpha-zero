from reversi_zero.config import Config
from reversi_zero.env.reversi_env import Player, ReversiEnv


class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.over = True
        self.human_is_black = None
        self.observers = []
        self.env = ReversiEnv().reset()

    def add_observer(self, observer_func):
        self.observers.append(observer_func)

    def start_game(self, human_is_black):
        self.human_is_black = human_is_black
        self.env = ReversiEnv().reset()

    @property
    def next_player(self):
        return self.env.next_player

    def stone(self, px, py):
        """left top=(0, 0), right bottom=(7,7)"""
        pos = int(py * 8 + px)
        assert 0 <= pos < 64
        bit = 1 << pos
        if self.env.board.black & bit:
            return Player.black
        elif self.env.board.white & bit:
            return Player.white
        return None

    def available(self, px, py, for_human=True):
        pos = int(py * 8 + px)
        if pos < 0 or 64 <= pos:
            return False

    def move(self, px, py, by_human=True):
        pos = int(py * 8 + px)
        assert 0 <= pos < 64

