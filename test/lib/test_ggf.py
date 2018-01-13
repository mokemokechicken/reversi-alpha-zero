import numpy as np

from nose.tools.trivial import eq_

from reversi_zero.lib.bitboard import board_to_string
from reversi_zero.lib.ggf import parse_ggf, convert_move_to_action, convert_action_to_move
from reversi_zero.lib.util import parse_ggf_board_to_bitboard

GGF_STR = '(;GM[Othello]PC[NBoard]DT[2014-02-21 20:52:27 GMT]PB[./mEdax]PW[chris]RE[?]TI[15:00]TY[8]' \
          'BO[8 --*O-----------------------O*------*O--------------------------- *]' \
          'B[F5]W[F6]B[D3]W[C5]B[E6]W[F7]B[E7]W[F4];)'


def test_parse_ggf():
    ggf = parse_ggf(GGF_STR)
    eq_("8", ggf.BO.board_type)
    eq_(64, len(ggf.BO.square_cont))
    eq_("*", ggf.BO.color)
    eq_(8, len(ggf.MOVES))
    eq_("B", ggf.MOVES[0].color)
    eq_("F5", ggf.MOVES[0].pos)
    eq_("W", ggf.MOVES[1].color)
    eq_("F6", ggf.MOVES[1].pos)


def test_parse_ggf_board_to_bitboard():
    ggf = parse_ggf(GGF_STR)
    black, white = parse_ggf_board_to_bitboard(ggf.BO.square_cont)
    eq_(EXPECTED1.strip(), board_to_string(black, white).strip())


def test_convert_move_to_action():
    eq_(0, convert_move_to_action("A1"))
    eq_(63, convert_move_to_action("H8"))
    eq_(44, convert_move_to_action("F5"))
    eq_(None, convert_move_to_action("PA"))


def test_convert_action_to_move():
    eq_("A1", convert_action_to_move(0))
    eq_("H8", convert_action_to_move(63))
    eq_("F5", convert_action_to_move(44))
    eq_("PA", convert_action_to_move(None))


EXPECTED1 = '''
##########
#  OX    #
#        #
#        #
#   XO   #
#   OX   #
#        #
#        #
#        #
##########
'''