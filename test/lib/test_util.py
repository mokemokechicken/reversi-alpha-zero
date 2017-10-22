from nose.tools.trivial import eq_

from reversi_zero.lib import util
from reversi_zero.lib.bitboard import board_to_string


def test_parse_to_bitboards_init():
    ex = '''
    ##########
    #        #
    #        #
    #        #
    #   OX   #
    #   XO   #
    #        #
    #        #
    #        #
    ##########
    '''

    black, white = util.parse_to_bitboards(ex)
    eq_(black, 0b00001000 << 24 | 0b00010000 << 32, f"{ex}\n-------\n{board_to_string(black, white)}")
    eq_(white, 0b00010000 << 24 | 0b00001000 << 32, f"{ex}\n-------\n{board_to_string(black, white)}")


def test_parse_to_bitboards():
    ex = '''
##########
#OO      #
#XOO     #
#OXOOO   #
#  XOX   #
#   XXX  #
#  X     #
# X      #
#       X#
##########'''

    black, white = util.parse_to_bitboards(ex)
    eq_(ex.strip(), board_to_string(black, white).strip(), f"{ex}\n-------\n{board_to_string(black, white)}")
