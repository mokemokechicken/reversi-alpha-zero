from time import time

from logging import getLogger

from reversi_zero.env.reversi_env import ReversiEnv, Player
from reversi_zero.lib.bitboard import find_correct_moves
import numpy as np


logger = getLogger(__name__)


class Timeout(Exception):
    pass


class ReversiSolver:
    """calculate which is winner. Not estimation by NN!

    this implementation runs very slow. (^^;
    """
    def __init__(self):
        self.cache = {}
        self.start_time = None
        self.timeout = None
        self.last_is_exactly = False

    def solve(self, black, white, next_player, timeout=30, exactly=False):
        self.timeout = timeout
        self.start_time = time()
        if not self.last_is_exactly and exactly:
            self.cache = {}
        self.last_is_exactly = exactly
        
        try:
            # logger.debug("start resolving")
            move, score = self.find_winning_move_and_score(ReversiEnv().update(black, white, next_player),
                                                           exactly=exactly)
            if next_player == Player.white:
                score = -score
            # logger.debug(f"solve answer=({move},{score})({time()-self.start_time:.3f} seconds)")
            return move, score
        except Timeout:
            return None, None

    def find_winning_move_and_score(self, env: ReversiEnv, exactly=True):
        if env.done:
            b, w = env.board.number_of_black_and_white
            return None, b - w
        if time() - self.start_time > self.timeout:
            logger.debug("timeout!")
            raise Timeout()

        turn = env.turn
        key = black, white, next_player = env.board.black, env.board.white, env.next_player
        if key in self.cache:
            return self.cache[key]

        if next_player == Player.black:
            legal_moves = find_correct_moves(black, white)
        else:
            legal_moves = find_correct_moves(white, black)

        action_list = [idx for idx in range(64) if legal_moves & (1 << idx)]
        score_list = np.zeros(len(action_list), dtype=int)
        for i, action in enumerate(action_list):
            # env.update(black, white, next_player)
            env.board.black = black
            env.board.white = white
            env.next_player = next_player
            env.turn = turn
            env.done = False
            env.winner = None
            #
            env.step(action)
            _, score = self.find_winning_move_and_score(env, exactly=exactly)
            score_list[i] = score

            if not exactly:
                # do not need to find the best score move
                if next_player == Player.black and score > 0:
                    break
                elif next_player == Player.white and score < 0:
                    break

        # print(list(zip(action_list, score_list)))

        if next_player == Player.black:
            best_action = action_list[int(np.argmax(score_list))]
            best_score = np.max(score_list)
        else:
            best_action = action_list[int(np.argmin(score_list))]
            best_score = np.min(score_list)

        self.cache[key] = (best_action, best_score)
        return best_action, best_score


if __name__ == '__main__':
    from reversi_zero.lib.util import parse_to_bitboards

    def q1():
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
        rr = ReversiSolver()
        print("correct is (57, +2)")
        print(rr.solve(b, w, Player.white, exactly=False))
        print(len(rr.cache))

    def q2():
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
        rr = ReversiSolver()
        print("correct is (4 or 14, -2)")
        print(rr.solve(b, w, Player.black, exactly=False))
        print(len(rr.cache))

    def q3():  # O: black, X: white
        board = '''
        ##########
        #  X OOO #
        #X XOXO O#
        #XXXXOXOO#
        #XOXOOXXO#
        #XOOOOXXO#
        #XOOOXXXO#
        # OOOOXX #
        #  OOOOX #
        ##########'''
        b, w = parse_to_bitboards(board)
        rr = ReversiSolver()
        print("correct is (3, +2)")
        print(rr.solve(b, w, Player.white, exactly=True))
        print(len(rr.cache))

    q3()

