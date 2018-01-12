# http://primenumber.hatenadiary.jp/entry/2016/12/26/063226
import numpy as np

BLACK_CHR = "O"
WHITE_CHR = "X"
EXTRA_CHR = "*"


def board_to_string(black, white, with_edge=True, extra=None):
    """
     0  1  2  3  4  5  6  7
     8  9 10 11 12 13 14 15
    ..
    56 57 58 59 60 61 62 63

    0: Top Left, LSB
    63: Bottom Right

    :param black: bitboard
    :param white: bitboard
    :param with_edge:
    :param extra: bitboard
    :return:
    """
    array = [" "] * 64
    extra = extra or 0
    for i in range(64):
        if black & 1:
            array[i] = BLACK_CHR
        elif white & 1:
            array[i] = WHITE_CHR
        elif extra & 1:
            array[i] = EXTRA_CHR
        black >>= 1
        white >>= 1
        extra >>= 1

    ret = ""
    if with_edge:
        ret = "#" * 10 + "\n"
    for y in range(8):
        if with_edge:
            ret += "#"
        ret += "".join(array[y * 8:y * 8 + 8])
        if with_edge:
            ret += "#"
        ret += "\n"
    if with_edge:
        ret += "#" * 10 + "\n"
    return ret


def find_correct_moves(own, enemy):
    """return legal moves"""
    left_right_mask = 0x7e7e7e7e7e7e7e7e  # Both most left-right edge are 0, else 1
    top_bottom_mask = 0x00ffffffffffff00  # Both most top-bottom edge are 0, else 1
    mask = left_right_mask & top_bottom_mask
    mobility = 0
    mobility |= search_offset_left(own, enemy, left_right_mask, 1)  # Left
    mobility |= search_offset_left(own, enemy, mask, 9)  # Left Top
    mobility |= search_offset_left(own, enemy, top_bottom_mask, 8)  # Top
    mobility |= search_offset_left(own, enemy, mask, 7)  # Top Right
    mobility |= search_offset_right(own, enemy, left_right_mask, 1)  # Right
    mobility |= search_offset_right(own, enemy, mask, 9)  # Bottom Right
    mobility |= search_offset_right(own, enemy, top_bottom_mask, 8)  # Bottom
    mobility |= search_offset_right(own, enemy, mask, 7)  # Left bottom
    return mobility


def calc_flip(pos, own, enemy):
    """return flip stones of enemy by bitboard when I place stone at pos.

    :param pos: 0~63
    :param own: bitboard (0=top left, 63=bottom right)
    :param enemy: bitboard
    :return: flip stones of enemy when I place stone at pos.
    """
    assert 0 <= pos <= 63, f"pos={pos}"
    f1 = _calc_flip_half(pos, own, enemy)
    f2 = _calc_flip_half(63 - pos, rotate180(own), rotate180(enemy))
    return f1 | rotate180(f2)


def _calc_flip_half(pos, own, enemy):
    el = [enemy, enemy & 0x7e7e7e7e7e7e7e7e, enemy & 0x7e7e7e7e7e7e7e7e, enemy & 0x7e7e7e7e7e7e7e7e]
    masks = [0x0101010101010100, 0x00000000000000fe, 0x0002040810204080, 0x8040201008040200]
    masks = [b64(m << pos) for m in masks]
    flipped = 0
    for e, mask in zip(el, masks):
        outflank = mask & ((e | ~mask) + 1) & own
        flipped |= (outflank - (outflank != 0)) & mask
    return flipped


def search_offset_left(own, enemy, mask, offset):
    e = enemy & mask
    blank = ~(own | enemy)
    t = e & (own >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)  # Up to six stones can be turned at once
    return blank & (t >> offset)  # Only the blank squares can be started


def search_offset_right(own, enemy, mask, offset):
    e = enemy & mask
    blank = ~(own | enemy)
    t = e & (own << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)  # Up to six stones can be turned at once
    return blank & (t << offset)  # Only the blank squares can be started


def flip_vertical(x):
    k1 = 0x00FF00FF00FF00FF
    k2 = 0x0000FFFF0000FFFF
    x = ((x >> 8) & k1) | ((x & k1) << 8)
    x = ((x >> 16) & k2) | ((x & k2) << 16)
    x = (x >> 32) | b64(x << 32)
    return x


def b64(x):
    return x & 0xFFFFFFFFFFFFFFFF


def bit_count(x):
    return bin(x).count('1')


def bit_to_array(x, size):
    """bit_to_array(0b0010, 4) -> array([0, 1, 0, 0])"""
    return np.array(list(reversed((("0" * size) + bin(x)[2:])[-size:])), dtype=np.uint8)


def flip_diag_a1h8(x):
    k1 = 0x5500550055005500
    k2 = 0x3333000033330000
    k4 = 0x0f0f0f0f00000000
    t = k4 & (x ^ b64(x << 28))
    x ^= t ^ (t >> 28)
    t = k2 & (x ^ b64(x << 14))
    x ^= t ^ (t >> 14)
    t = k1 & (x ^ b64(x << 7))
    x ^= t ^ (t >> 7)
    return x


def rotate90(x):
    return flip_diag_a1h8(flip_vertical(x))


def rotate180(x):
    return rotate90(rotate90(x))


def dirichlet_noise_of_mask(mask, alpha):
    num_1 = bit_count(mask)
    noise = list(np.random.dirichlet([alpha] * num_1))
    ret_list = []
    for i in range(64):
        if (1 << i) & mask:
            ret_list.append(noise.pop(0))
        else:
            ret_list.append(0)
    return np.array(ret_list)
