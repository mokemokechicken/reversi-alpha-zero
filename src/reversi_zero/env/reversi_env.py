import enum

from logging import getLogger

from reversi_zero.lib.bitboard import board_to_string, calc_flip, bit_count, find_correct_moves

logger = getLogger(__name__)
# noinspection PyArgumentList
Player = enum.Enum("Player", "black white")
# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")


def another_player(player: Player):
    return Player.white if player == Player.black else Player.black


class ReversiEnv:
    def __init__(self):
        self.board = None
        self.next_player = None  # type: Player
        self.turn = 0
        self.done = False
        self.winner = None  # type: Winner

    def reset(self):
        self.board = Board()
        self.next_player = Player.black
        self.turn = 0
        self.done = False
        self.winner = None
        return self

    def update(self, black, white, next_player):
        self.board = Board(black, white)
        self.next_player = next_player
        self.turn = sum(self.board.number_of_black_and_white) - 4
        self.done = False
        self.winner = None
        return self

    def step(self, action):
        """

        :param int|None action: move pos=0 ~ 63 (0=top left, 7 top right, 63 bottom right), None is resign
        :return:
        """
        assert action is None or 0 <= action <= 63, f"Illegal action={action}"

        if action is None:
            self._resigned()
            return self.board, {}

        own, enemy = self.get_own_and_enemy()

        flipped = calc_flip(action, own, enemy)
        if bit_count(flipped) == 0:
            self.illegal_move_to_lose(action)
            return self.board, {}
        own ^= flipped
        own |= 1 << action
        enemy ^= flipped

        self.set_own_and_enemy(own, enemy)
        self.turn += 1

        if bit_count(find_correct_moves(enemy, own)) > 0:  # there are legal moves for enemy.
            self.change_to_next_player()
        elif bit_count(find_correct_moves(own, enemy)) > 0:  # there are legal moves for me but enemy.
            pass
        else:  # there is no legal moves for me and enemy.
            self._game_over()

        return self.board, {}

    def _game_over(self):
        self.done = True
        if self.winner is None:
            black_num, white_num = self.board.number_of_black_and_white
            if black_num > white_num:
                self.winner = Winner.black
            elif black_num < white_num:
                self.winner = Winner.white
            else:
                self.winner = Winner.draw

    def change_to_next_player(self):
        self.next_player = another_player(self.next_player)

    def illegal_move_to_lose(self, action):
        logger.warning(f"Illegal action={action}, No Flipped!")
        self._win_another_player()
        self._game_over()

    def _resigned(self):
        self._win_another_player()
        self._game_over()

    def _win_another_player(self):
        win_player = another_player(self.next_player)  # type: Player
        if win_player == Player.black:
            self.winner = Winner.black
        else:
            self.winner = Winner.white

    def get_own_and_enemy(self):
        if self.next_player == Player.black:
            own, enemy = self.board.black, self.board.white
        else:
            own, enemy = self.board.white, self.board.black
        return own, enemy

    def set_own_and_enemy(self, own, enemy):
        if self.next_player == Player.black:
            self.board.black, self.board.white = own, enemy
        else:
            self.board.white, self.board.black = own, enemy

    def render(self):
        b, w = self.board.number_of_black_and_white
        print(f"next={self.next_player.name} turn={self.turn} B={b} W={w}")
        print(board_to_string(self.board.black, self.board.white, with_edge=True))

    @property
    def observation(self):
        """

        :rtype: Board
        """
        return self.board


class Board:
    def __init__(self, black=None, white=None, init_type=0):
        self.black = black or (0b00010000 << 24 | 0b00001000 << 32)
        self.white = white or (0b00001000 << 24 | 0b00010000 << 32)

        if init_type:
            self.black, self.white = self.white, self.black

    @property
    def number_of_black_and_white(self):
        return bit_count(self.black), bit_count(self.white)

