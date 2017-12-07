from nose.tools.trivial import eq_, ok_

import numpy as np


from reversi_zero.config import Config
from reversi_zero.agent.player import ReversiPlayer
from reversi_zero.lib.bitboard import bit_count


def test_add_data_to_move_buffer_with_8_symmetries():
    config = Config()
    player = ReversiPlayer(config, None)

    """
    board: p=0.2, q=0.8, O=own, X=enemy
     01234567 - x
    0O      q
    1 O
    2
    3
    4
    5
    6       X
    7p      X
    |
    y
    """

    own = stone_bit(0, 0) | stone_bit(1, 1)
    enemy = stone_bit(7, 6) | stone_bit(7, 7)
    policy = np.zeros((64, ))
    policy[idx(7, 0)] = 0.8
    policy[idx(0, 7)] = 0.2
    player.add_data_to_move_buffer_with_8_symmetries(own, enemy, policy)

    # no transform
    (o, e), p = player.moves[0]  # own, enemy, policy
    eq_((bit_count(o), bit_count(e)), (2, 2))
    ok_(check_bit(o, 0, 0))
    ok_(check_bit(o, 1, 1))
    ok_(check_bit(e, 7, 6))
    ok_(check_bit(e, 7, 7))
    eq_(p[idx(7, 0)], 0.8)
    eq_(p[idx(0, 7)], 0.2)

    # rotate right
    (o, e), p = player.moves[1]  # own, enemy, policy
    eq_((bit_count(o), bit_count(e)), (2, 2))
    ok_(check_bit(o, 7, 0))
    ok_(check_bit(o, 6, 1))
    ok_(check_bit(e, 0, 7))
    ok_(check_bit(e, 1, 7))
    eq_(p[idx(7, 7)], 0.8)
    eq_(p[idx(0, 0)], 0.2)

    # rotate right twice
    (o, e), p = player.moves[2]  # own, enemy, policy
    eq_((bit_count(o), bit_count(e)), (2, 2))
    ok_(check_bit(o, 7, 7))
    ok_(check_bit(o, 6, 6))
    ok_(check_bit(e, 0, 0))
    ok_(check_bit(e, 0, 1))
    eq_(p[idx(0, 7)], 0.8)
    eq_(p[idx(7, 0)], 0.2)

    # flip vertical -> rotate right
    (o, e), p = player.moves[5]  # own, enemy, policy
    eq_((bit_count(o), bit_count(e)), (2, 2))
    ok_(check_bit(o, 0, 0))
    ok_(check_bit(o, 1, 1))
    ok_(check_bit(e, 6, 7))
    ok_(check_bit(e, 7, 7))
    eq_(p[idx(0, 7)], 0.8)
    eq_(p[idx(7, 0)], 0.2)


def idx(x, y):
    return y*8 + x


def stone_bit(x, y):
    return 1 << idx(x, y)


def check_bit(bb, x, y):
    return bb & stone_bit(x, y) != 0

