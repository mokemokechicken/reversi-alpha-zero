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


cdef Env create_env(unsigned long long own, unsigned long long enemy, int next_player):
    legal_moves = find_correct_moves(own, enemy)
    return Env(own, enemy, next_player, legal_moves, 0, -1, best_move=-1, best_score=-100, next_search_action=0)


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
        self.timeout = int(timeout)
        self.start_time = time()
        if not self.last_is_exactly and exactly:
            self.cache = {}
        self.last_is_exactly = exactly

        result = self.find_winning_move_and_score(black, white, next_player, exactly)
        return result

    cdef SolveResult find_winning_move_and_score(self, unsigned long long black, unsigned long long white, int next_player, int exactly):
        cdef Env *child_env
        cdef Env *env
        cdef Env next_env
        cdef EnvStack stack = EnvStack()
        cdef unsigned long long next_own, next_enemy

        if next_player == BLACK:
            root_env = create_env(black, white, next_player)
        else:
            root_env = create_env(white, black, next_player)

        stack.add(&root_env)

        while stack.size():
            env = stack.top()
            print(env.enemy)
            if child_env and child_env.finished:
                if env.best_score < child_env.best_score:
                    env.best_move = child_env.parent_action
                    env.best_score = child_env.best_score
            child_env = NULL

            action = next_action(env)
            if action == -1:
                child_env = stack.pop()
                child_env.finished = 1
                continue

            flipped = calc_flip(action, env.own, env.enemy)

            next_own = (env.own ^ flipped) | (1 << action)
            next_enemy = env.enemy ^ flipped
            if find_correct_moves(next_enemy, next_own) > 0:
                next_env = create_env(next_enemy, next_own, 3 - env.next_player)  # next_player: 2 => 1, 1 => 2
                next_env.parent_action = action
                stack.add(&next_env)
            elif find_correct_moves(next_own, next_enemy) > 0:
                next_env = create_env(next_own, next_enemy, env.next_player)
                next_env.parent_action = action
                stack.add(&next_env)
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
    cdef Env* stack[64]
    cdef int pos

    def __init__(self):
        self.pos = 0

    cdef add(self, Env *env):
        self.stack[self.pos] = env
        self.pos += 1

    cdef Env* top(self):
        return self.stack[self.pos - 1]

    cdef Env* pop(self):
        cdef Env *ret = self.top()
        self.stack[self.pos] = NULL
        self.pos -= 1
        return ret

    cdef int size(self):
        return self.pos


