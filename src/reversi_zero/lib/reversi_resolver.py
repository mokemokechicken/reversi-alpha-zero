from reversi_zero.env.reversi_env import ReversiEnv, Player
from reversi_zero.lib.bitboard import find_correct_moves
import numpy as np


class ReversiResolver:
    def __init__(self):
        self.cache = {}

    def resolve(self, env: ReversiEnv):
        # print(f"start resolve from turn={env.turn}")
        move, score = self.find_best_move_and_score(ReversiEnv().update(env.board.black, env.board.white, env.next_player))
        if env.next_player == Player.white:
            score = -score
        return move, score

    def find_best_move_and_score(self, env: ReversiEnv):
        if env.done:
            b, w = env.board.number_of_black_and_white
            return None, b - w
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
            _, score = self.find_best_move_and_score(env)
            score_list[i] = score
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
        e = ReversiEnv().update(b, w, Player.white)
        rr = ReversiResolver()
        print("correct is (57, +2)")
        print(rr.resolve(e))
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
        e = ReversiEnv().update(b, w, Player.black)
        rr = ReversiResolver()
        print("correct is (4 or 14, -2)")
        print(rr.resolve(e))
        print(len(rr.cache))

    q1()

