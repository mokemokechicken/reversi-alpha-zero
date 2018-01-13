def parse_to_bitboards(string: str):
    lines = string.strip().split("\n")
    black = 0
    white = 0
    y = 0

    for line in [l.strip() for l in lines]:
        if line[:2] == '##':
            continue
        for i, ch in enumerate(line[1:9]):
            if ch == 'O':
                black |= 1 << (y*8+i)
            elif ch == 'X':
                white |= 1 << (y*8+i)
        y += 1

    return black, white


def parse_ggf_board_to_bitboard(string: str):
    white = black = 0
    for i, ch in enumerate(string):
        if ch == "*":
            black |= 1 << i
        elif ch == "O":
            white |= 1 << i
    return black, white
