import re
from collections import namedtuple

from reversi_zero.lib.util import parse_ggf_board_to_bitboard

GGF = namedtuple("GGF", "BO MOVES")
BO = namedtuple("BO", "board_type, square_cont, color")  # color: {O, *}  (O is white, * is black)
MOVE = namedtuple("MOVE", "color pos")  # color={B, W} pos: like 'F5'


def parse_ggf(ggf):
    """https://skatgame.net/mburo/ggsa/ggf

    :param ggf:
    :rtype: GGF
    """
    tokens = re.split(r'([a-zA-Z]+\[[^\]]+\])', ggf)
    moves = []
    bo = None
    for token in tokens:
        match = re.search(r'([a-zA-Z]+)\[([^\]]+)\]', token)
        if not match:
            continue
        key, value = re.search(r'([a-zA-Z]+)\[([^\]]+)\]', token).groups()
        key = key.upper()
        if key == "BO":
            bo = BO(*value.split(" "))
        elif key in ("B", "W"):
            moves.append(MOVE(key, value))
    return GGF(bo, moves)


def convert_move_to_action(move_str: str):
    """

    :param move_str: A1 -> 0, H8 -> 63
    :return:
    """
    if move_str[:2].lower() == "pa":
        return None
    pos = move_str.lower()
    y = ord(pos[0]) - ord("a")
    x = int(pos[1]) - 1
    return y * 8 + x


def convert_action_to_move(action: int):
    if action is None:
        return "PA"
    y = action // 8
    x = action % 8
    return chr(ord("A") + y) + str(x + 1)


def convert_to_bitboard_and_actions(ggf: GGF):
    black, white = parse_ggf_board_to_bitboard(ggf.BO.square_cont)
    actions = []
    for move in ggf.MOVES:  # type: MOVE
        actions.append(convert_move_to_action(move.pos))
    return black, white, actions
