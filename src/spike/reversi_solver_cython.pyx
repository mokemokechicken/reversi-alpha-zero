from time import time

from .bitboard_cython import bit_count, find_correct_moves, calc_flip

# CONST
DEF BLACK = 1
DEF WHITE = 2

# TYPE
cdef struct SolveResult:
    int move
    int score

cdef struct Env:
    unsigned long long own
    unsigned long long enemy
    int next_player
    unsigned long long legal_moves
    int finished
    int parent_action
    int best_move
    int best_score
    int next_search_action


# cdef Env create_env(unsigned long long own, unsigned long long enemy, int next_player):
#     legal_moves = find_correct_moves(own, enemy)
#     return Env(own, enemy, next_player, legal_moves, 0, -1, best_move=-1, best_score=-100, next_search_action=0)
#

cdef class ReversiSolver:
    cdef dict cache
    cdef float start_time
    cdef int timeout
    cdef int last_is_exactly

    def __init__(self):
        self.cache = {}
        self.start_time = 0
        self.timeout = 0
        self.last_is_exactly = 0

    def solve(self, black, white, next_player: int, timeout=30, exactly=False):
        """

        :param black:
        :param white:
        :param next_player: 1=Black, 2=White
        :param timeout:
        :param exactly:
        :return:
        """
        self.timeout = int(timeout)
        self.start_time = time()
        if not self.last_is_exactly and exactly:
            self.cache = {}
        self.last_is_exactly = exactly

        result = self.find_winning_move_and_score(black, white, next_player, exactly)
        if result.move < 0:
            return None, None
        else:
            return result.move, result.score

    cdef SolveResult find_winning_move_and_score(self, unsigned long long black, unsigned long long white, int next_player, int exactly):
        cdef Env* child_env = NULL
        cdef Env* env
        cdef Env* next_env
        cdef EnvStack stack = EnvStack()
        cdef unsigned long long next_own, next_enemy
        cdef int root_next_player = next_player
        cdef Env* root_env

        if next_player == BLACK:
            root_env = stack.add(black, white, next_player)
        else:
            root_env = stack.add(white, black, next_player)

        while stack.size():
            if time() - self.start_time > self.timeout:
                return SolveResult(-1, -100)

            env = stack.top()

            cache_key = (env.own, env.enemy, env.next_player)
            if cache_key in self.cache:
                obj = <SolveResult>self.cache[cache_key]
                env.best_move = obj.move
                env.best_score = obj.score
                child_env = stack.pop()
                child_env.finished = 1
                continue

            if child_env and child_env.finished:
                child_env.finished = 0
                child_score = child_env.best_score
                if child_env.next_player != env.next_player:
                    child_score = -child_score
                if env.best_score < child_score:
                    # print(("=" * stack.size()) + f"> update best score ({child_env.parent_action},{child_score})")
                    env.best_move = child_env.parent_action
                    env.best_score = child_score

            action = next_action(env)
            # print(("=" * stack.size()) + f"> ID={env.id} NP={env.next_player}:{action} best=({env.best_move},{env.best_score})")
            if action == -1 or (not exactly and env.best_score > 0):
                child_env = stack.pop()
                child_env.finished = 1
                self.cache[cache_key] = SolveResult(move=env.best_move, score=env.best_score)
                continue

            flipped = calc_flip(action, env.own, env.enemy)
            next_own = (env.own ^ flipped) | (1 << action)
            next_enemy = env.enemy ^ flipped

            if find_correct_moves(next_enemy, next_own) > 0:
                next_env = stack.add(next_enemy, next_own, 3 - env.next_player)  # next_player: 2 => 1, 1 => 2
                next_env.parent_action = action
            elif find_correct_moves(next_own, next_enemy) > 0:
                next_env = stack.add(next_own, next_enemy, env.next_player)
                next_env.parent_action = action
            else:
                score = bit_count(next_own) - bit_count(next_enemy)
                if env.best_score < score:
                    env.best_move = action
                    env.best_score = score

        return SolveResult(root_env.best_move, root_env.best_score)


cdef int next_action(Env *env):
    for i in range(env.next_search_action, 64):
        if env.legal_moves & (1 << i):
            env.next_search_action = i+1
            return i
    return -1


cdef class EnvStack:
    cdef Env stack[64]
    cdef int pos

    def __init__(self):
        self.pos = 0

    cdef Env* add(self, unsigned long long own, unsigned long long enemy, int next_player):
        cdef Env* env = &self.stack[self.pos]
        env.own = own
        env.enemy = enemy
        env.next_player = next_player
        env.legal_moves = find_correct_moves(own, enemy)
        env.finished = 0
        env.parent_action = -1
        env.best_move = -1
        env.best_score = -100
        env.next_search_action = 0
        self.pos += 1
        return env

    cdef Env* top(self):
        return &(self.stack[self.pos - 1])

    cdef Env* pop(self):
        ret = self.top()
        self.pos -= 1
        # print(f"pop: {ret.id} from pos: {self.pos}")
        return ret

    cdef int size(self):
        return self.pos

    cdef debug(self):
        for i in range(self.pos):
            print(f"{i}: {self.stack[i].id}")
