from time import time

import pyximport

from reversi_zero.env.reversi_env import Player
from reversi_zero.lib.util import parse_to_bitboards

pyximport.install()

import timeit


def examples():
    ret = [
    '''
    ##########
    #OO      #
    #XOO     #
    #OXOOO   #
    #  XOX   #
    #   XXX  #
    #  X     #
    # X      #
    #        #
    ##########
    ''',
    '''
    ##########
    #OOOOOXO #
    #OOOOOXOO#
    #OOOOOXOO#
    #OXOXOXOO#
    #OOXOXOXO#
    #OOOOOOOO#
    #XXXO   O#
    #        #
    ##########
    ''',
    '''
    ##########
    #OOXXXXX #
    #XOXXXXXX#
    #XXXXXXXX#
    #XOOXXXXX#
    #OXXXOOOX#
    #OXXOOOOX#
    #OXXXOOOX#
    # OOOOOOO#
    ##########
    ''']
    return ret


def test_find_correct_move():
    import spike.bitboard_cython as f
    import reversi_zero.lib.bitboard as b

    for ex in examples():
        black, white = parse_to_bitboards(ex)
        assert f.find_correct_moves(black, white) == b.find_correct_moves(black, white)
        cy = timeit.timeit("f.find_correct_moves(black, white)", globals=locals(), number=10000)
        py = timeit.timeit("b.find_correct_moves(black, white)", globals=locals(), number=10000)
        print(f"Cython={cy} : cPython={py}")


def test_calc_flip():
    import spike.bitboard_cython as f
    import reversi_zero.lib.bitboard as b

    for ex in examples():
        black, white = parse_to_bitboards(ex)
        assert f.find_correct_moves(black, white) == b.find_correct_moves(black, white)
        legal_moves = f.find_correct_moves(black, white)
        action_list = [idx for idx in range(64) if legal_moves & (1 << idx)]

        for action in action_list:
            assert f.calc_flip(action, black, white) == b.calc_flip(action, black, white)
            cy = timeit.timeit("f.calc_flip(action, black, white)", globals=locals(), number=10000)
            py = timeit.timeit("b.calc_flip(action, black, white)", globals=locals(), number=10000)
            print(f"Cython={cy} : cPython={py}")


def test_solve():
    def q1():
        import reversi_zero.lib.reversi_solver as p
        import spike.reversi_solver_cython as c
        board = '''
        ##########
        #XXXX    #
        #XOXX    #
        #XOXXOOOO#
        #XOXOXOOO#
        #XOXXOXOO#
        #OOOOXOXO#
        # OOOOOOO#
        #  XXXXXO#
        ##########'''
        b, w = parse_to_bitboards(board)
        print("correct is (57, +2)")

        start_time = time()
        ret = p.ReversiSolver().solve(b, w, next_player=Player.white, exactly=False)
        print(f"{time()-start_time} sec: ret={ret}")

        start_time = time()
        ret = c.ReversiSolver().solve(b, w, next_player=2, exactly=False)
        print(f"{time()-start_time} sec: ret={ret}")

        # rr = p.ReversiSolver()
        # print(rr.solve(b, w, Player.white, exactly=False))
        # print(len(rr.cache))

    def q2():
        import reversi_zero.lib.reversi_solver as p
        import spike.reversi_solver_cython as c
        board = '''
        ##########
        #XXXX    #
        #XXXX X  #
        #XXXXXXOO#
        #XXXXXXOO#
        #XXXXOXOO#
        #OXOOXOXO#
        # OOOOOOO#
        #OOOOOOOO#
        ##########'''
        b, w = parse_to_bitboards(board)

        start_time = time()
        ret = p.ReversiSolver().solve(b, w, next_player=Player.black, exactly=True)
        print(f"{time()-start_time} sec: ret={ret}")

        start_time = time()
        ret = c.ReversiSolver().solve(b, w, next_player=1, exactly=True)
        print(f"{time()-start_time} sec: ret={ret}")

    def q3():
        import reversi_zero.lib.reversi_solver as p
        import spike.reversi_solver_cython as c
        board = '''
        ##########
        #XXXXOOOX#
        #XXXX XOX#
        #XXXXXXOO#
        #XXXXXOO #
        #XXXXOXOO#
        #OXOOXOXO#
        # OOOOOOO#
        #OOOOOOOO#
        ##########'''
        b, w = parse_to_bitboards(board)
        start_time = time()
        print(p.ReversiSolver().solve(b, w, next_player=Player.black, exactly=True))
        ret = c.ReversiSolver().solve(b, w, next_player=1, exactly=True)
        print(f"{time()-start_time} sec: ret={ret}")

    def q4():
        import reversi_zero.lib.reversi_solver as p
        import spike.reversi_solver_cython as c
        board = '''
        ##########
        # X  XXXO#
        #O XXXXXX#
        #OOXOOOXX#
        #OOOOOOXO#
        #OOOOXOOO#
        #OOOOXOOX#
        # XXXOO  #
        #   OOO  #
        ##########'''
        b, w = parse_to_bitboards(board)
        start_time = time()
        ret = p.ReversiSolver().solve(b, w, next_player=Player.black, exactly=True)
        print(f"{time()-start_time} sec: ret={ret}")

        start_time = time()
        ret = c.ReversiSolver().solve(b, w, next_player=1, exactly=True)
        print(f"{time()-start_time} sec: ret={ret}")

    q4()


def test_bitcount():
    import spike.bitboard_cython as c
    import reversi_zero.lib.bitboard as p

    x = 4242342758
    assert p.bit_count(x) == c.bc_timeit(x)
    print(timeit.timeit("p.bit_count(x)", number=100000, globals=locals()))
    print(timeit.timeit("c.bc_timeit(x)", number=1, globals=locals()))


if __name__ == '__main__':
    # print("find_correct_moves")
    # test_find_correct_move()
    # print("calc_flip")
    # test_calc_flip()
    test_solve()
    #test_bitcount()
