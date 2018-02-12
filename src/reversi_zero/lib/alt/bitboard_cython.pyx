cpdef unsigned long long find_correct_moves(unsigned long long own, unsigned long long enemy):
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


cpdef unsigned long long calc_flip(int pos, unsigned long long own, unsigned long long enemy):
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


cdef inline unsigned long long _calc_flip_half(int pos, unsigned long long own, unsigned long long enemy):
    el = [enemy, enemy & 0x7e7e7e7e7e7e7e7e, enemy & 0x7e7e7e7e7e7e7e7e, enemy & 0x7e7e7e7e7e7e7e7e]
    masks = [0x0101010101010100, 0x00000000000000fe, 0x0002040810204080, 0x8040201008040200]
    masks = [(m << pos) for m in masks]
    flipped = 0
    for e, mask in zip(el, masks):
        outflank = mask & ((e | ~mask) + 1) & own
        flipped |= (outflank - (outflank != 0)) & mask
    return flipped


cdef inline unsigned long long flip_vertical(unsigned long long x):
    k1 = 0x00FF00FF00FF00FF
    k2 = 0x0000FFFF0000FFFF
    x = ((x >> 8) & k1) | ((x & k1) << 8)
    x = ((x >> 16) & k2) | ((x & k2) << 16)
    x = (x >> 32) | (x << 32)
    return x


cdef inline unsigned long long flip_diag_a1h8(unsigned long long x):
    k1 = 0x5500550055005500
    k2 = 0x3333000033330000
    k4 = 0x0f0f0f0f00000000
    t = k4 & (x ^ (x << 28))
    x ^= t ^ (t >> 28)
    t = k2 & (x ^ (x << 14))
    x ^= t ^ (t >> 14)
    t = k1 & (x ^ (x << 7))
    x ^= t ^ (t >> 7)
    return x


cdef inline unsigned long long rotate90(unsigned long long x):
    return flip_diag_a1h8(flip_vertical(x))


cdef inline unsigned long long rotate180(unsigned long long x):
    return rotate90(rotate90(x))


cdef inline unsigned long long search_offset_left(unsigned long long own, unsigned long long enemy, unsigned long long mask, int offset):
    e = enemy & mask
    blank = ~(own | enemy)
    t = e & (own >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)
    t |= e & (t >> offset)  # Up to six stones can be turned at once
    return blank & (t >> offset)  # Only the blank squares can be started


cdef inline unsigned long long search_offset_right(unsigned long long own, unsigned long long enemy, unsigned long long mask, int offset):
    e = enemy & mask
    blank = ~(own | enemy)
    t = e & (own << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)
    t |= e & (t << offset)  # Up to six stones can be turned at once
    return blank & (t << offset)  # Only the blank squares can be started


cpdef int bit_count(unsigned long long x):
    cdef int ret = 0
    for i in range(64):
        ret += x & 1
        x = x >> 1
    return ret

