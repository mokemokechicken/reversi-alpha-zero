import numpy as np

from nose.tools import assert_almost_equal
from nose.tools.trivial import ok_, eq_

from reversi_zero.lib.bitboard import find_correct_moves, board_to_string, bit_count, dirichlet_noise_of_mask, \
    bit_to_array
from reversi_zero.lib.util import parse_to_bitboards


def test_find_correct_moves_1():
    ex = '''
##########
#OO      #
#XOO     #
#OXOOO   #
#  XOX   #
#   XXX  #
#  X     #
# X      #
#        #
##########'''

    expect = '''
##########
#OO      #
#XOO     #
#OXOOO   #
#**XOX*  #
# **XXX  #
#  X**** #
# X      #
#        #
##########
'''
    _flip_test(ex, expect)


def _flip_test(ex, expect, player_black=True):
    b, w = parse_to_bitboards(ex)
    moves = find_correct_moves(b, w) if player_black else find_correct_moves(w, b)
    res = board_to_string(b, w, extra=moves)
    eq_(res.strip(), expect.strip(), f"\n{res}----{expect}")


def test_find_correct_moves_2():
    ex = '''
##########
#OOOOOXO #
#OOOOOXOO#
#OOOOOXOO#
#OXOXOXOO#
#OOXOXOXO#
#OOOOOOOO#
#XXXO   O#
#        #
##########'''

    expect = '''
##########
#OOOOOXO*#
#OOOOOXOO#
#OOOOOXOO#
#OXOXOXOO#
#OOXOXOXO#
#OOOOOOOO#
#XXXO***O#
#   *    #
##########'''

    _flip_test(ex, expect, player_black=False)


def test_find_correct_moves_3():
    ex = '''
##########
#OOXXXXX #
#XOXXXXXX#
#XXXXXXXX#
#XOOXXXXX#
#OXXXOOOX#
#OXXOOOOX#
#OXXXOOOX#
# OOOOOOO#
##########'''

    expect1 = '''
##########
#OOXXXXX #
#XOXXXXXX#
#XXXXXXXX#
#XOOXXXXX#
#OXXXOOOX#
#OXXOOOOX#
#OXXXOOOX#
#*OOOOOOO#
##########'''

    expect2 = '''
##########
#OOXXXXX*#
#XOXXXXXX#
#XXXXXXXX#
#XOOXXXXX#
#OXXXOOOX#
#OXXOOOOX#
#OXXXOOOX#
# OOOOOOO#
##########'''

    _flip_test(ex, expect1, player_black=False)
    _flip_test(ex, expect2, player_black=True)


def test_dirichlet_noise_of_mask():
    legal_moves = 47289423
    bc = bit_count(legal_moves)
    noise = dirichlet_noise_of_mask(legal_moves, 0.5)
    assert_almost_equal(1, np.sum(noise))
    eq_(bc, np.sum(noise > 0))
    ary = bit_to_array(legal_moves, 64)
    eq_(list(noise), list(noise * ary))
