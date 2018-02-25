import re
import sys
from collections import namedtuple

from logging import getLogger, StreamHandler, FileHandler
from time import time

from reversi_zero.agent.player import ReversiPlayer, CallbackInMCTS
from reversi_zero.config import Config, PlayWithHumanConfig
from reversi_zero.env.reversi_env import ReversiEnv, Player
from reversi_zero.lib.ggf import parse_ggf, convert_to_bitboard_and_actions, convert_move_to_action, \
    convert_action_to_move
from reversi_zero.lib.nonblocking_stream_reader import NonBlockingStreamReader
from reversi_zero.play_game.common import load_model

logger = getLogger(__name__)

GameState = namedtuple("GameState", "black white actions player")
GoResponse = namedtuple("GoResponse", "action eval time")
HintResponse = namedtuple("HintResponse", "action value visit")


def start(config: Config):
    config.play_with_human.update_play_config(config.play)
    root_logger = getLogger()
    for h in root_logger.handlers:
        if isinstance(h, StreamHandler) and not isinstance(h, FileHandler):
            root_logger.removeHandler(h)
    logger.info(f"config type={config.type}")
    NBoardEngine(config).start()
    logger.info("finish nboard")


class NBoardEngine:
    def __init__(self, config: Config):
        self.config = config
        self.reader = NonBlockingStreamReader(sys.stdin)
        self.handler = NBoardProtocolVersion2(config, self)
        self.running = False
        self.nc = self.config.nboard  # shorcut
        #
        self.env = ReversiEnv().reset()
        self.model = load_model(self.config)
        self.play_config = self.config.play
        self.player = self.create_player()
        self.turn_of_nboard = None

    def create_player(self):
        logger.debug("create new ReversiPlayer()")
        return ReversiPlayer(self.config, self.model, self.play_config, enable_resign=False)

    def start(self):
        self.running = True
        self.reader.start(push_callback=self.push_callback)
        while self.running and not self.reader.closed:
            message = self.reader.readline(self.nc.read_stdin_timeout)
            if message is None:
                continue
            message = message.strip()
            logger.debug(f"> {message}")
            self.handler.handle_message(message)

    def push_callback(self, message: str):
        # note: called in another thread
        if message.startswith("ping"):  # interupt
            self.stop_thinkng()

    def stop(self):
        self.running = False

    def reply(self, message):
        logger.debug(f"< {message}")
        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    def stop_thinkng(self):
        self.player.stop_thinking()

    def set_depth(self, n):
        try:
            n = int(n)
            # self.play_config.simulation_num_per_move = n * self.nc.simulation_num_per_depth_about
            self.play_config.required_visit_to_decide_action = n * self.nc.simulation_num_per_depth_about
            self.play_config.thinking_loop = min(
                30,
                int(self.play_config.required_visit_to_decide_action * 5 / self.play_config.simulation_num_per_move)
            )

            logger.info(f"set required_visit_to_decide_action to {self.play_config.required_visit_to_decide_action}")
        except ValueError:
            pass

    def reset_state(self):
        self.player = self.create_player()

    def set_game(self, game_state: GameState):
        self.env.reset()
        self.env.update(game_state.black, game_state.white, game_state.player)
        self.turn_of_nboard = game_state.player
        for action in game_state.actions:
            self._change_turn()
            if action is not None:
                self.env.step(action)

    def _change_turn(self):
        if self.turn_of_nboard:
            self.turn_of_nboard = Player.black if self.turn_of_nboard == Player.white else Player.white

    def move(self, action):
        self._change_turn()
        if action is not None:
            self.env.step(action)

    def go(self) -> GoResponse:
        if self.env.next_player != self.turn_of_nboard:
            return GoResponse(None, 0, 0)

        board = self.env.board
        if self.env.next_player == Player.black:
            states = (board.black, board.white)
        else:
            states = (board.white, board.black)
        start_time = time()
        action = self.player.action(*states)
        item = self.player.ask_thought_about(*states)
        evaluation = item.values[action]
        time_took = time() - start_time
        return GoResponse(action, evaluation, time_took)

    def hint(self, n_hint):
        """

        :param n_hint:
        """
        board = self.env.board
        if self.env.next_player == Player.black:
            states = (board.black, board.white)
        else:
            states = (board.white, board.black)

        def hint_report_callback(values, visits):
            hint_list = []
            for action, visit in list(sorted(enumerate(visits), key=lambda x: -x[1]))[:n_hint]:
                if visit > 0:
                    hint_list.append(HintResponse(action, values[action], visit))
            self.handler.report_hint(hint_list)

        callback_info = CallbackInMCTS(self.config.nboard.hint_callback_per_sim, hint_report_callback)
        self.player.action(*states, callback_in_mtcs=callback_info)
        item = self.player.ask_thought_about(*states)
        hint_report_callback(item.values, item.visit)


class NBoardProtocolVersion2:
    def __init__(self, config: Config, engine: NBoardEngine):
        self.config = config
        self.engine = engine
        self.handlers = [
            (re.compile(r'nboard ([0-9]+)'), self.nboard),
            (re.compile(r'set depth ([0-9]+)'), self.set_depth),
            (re.compile(r'set game (.+)'), self.set_game),
            (re.compile(r'move ([^/]+)(/[^/]*)?(/[^/]*)?'), self.move),
            (re.compile(r'hint ([0-9]+)'), self.hint),
            (re.compile(r'go'), self.go),
            (re.compile(r'ping ([0-9]+)'), self.ping),
            (re.compile(r'learn'), self.learn),
            (re.compile(r'analyze'), self.analyze),
        ]

    def handle_message(self, message):
        for regexp, func in self.handlers:
            if self.scan(message, regexp, func):
                return
        logger.debug(f"ignore message: {message}")

    def scan(self, message, regexp, func):
        match = regexp.match(message)
        if match:
            func(*match.groups())
            return True
        return False

    def nboard(self, version):
        if version != "2":
            logger.warning(f"UNKNOWN NBoard Version {version}!!!")
        self.engine.reply(f"set myname {self.config.nboard.my_name}({self.config.type})")
        self.tell_status("waiting")

    def set_depth(self, depth):
        """Set engine midgame search depth.

        Optional: Set midgame depth to {maxDepth}. Endgame depths are at the engine author's discretion.
        :param depth:
        """
        self.engine.set_depth(depth)

    def set_game(self, ggf_str):
        """Tell the engine that all further commands relate to the position at the end of the given game, in GGF format.

        Required:The engine must update its stored game state.
        :param ggf_str: see https://skatgame.net/mburo/ggsa/ggf . important info are BO, B+, W+
        """
        ggf = parse_ggf(ggf_str)
        black, white, actions = convert_to_bitboard_and_actions(ggf)
        player = Player.black if ggf.BO.color == "*" else Player.white
        self.engine.set_game(GameState(black, white, actions, player))

        # if set_game at turn=1~2 is sent, reset engine state.
        if len(actions) <= 1:
            self.engine.reset_state()  # clear MCTS cache

    def move(self, move, evaluation, time_sec):
        """Tell the engine that all further commands relate to the position after the given move.
        The move is 2 characters e.g. "F5". Eval is normally in centi-disks. Time is in seconds.
        Eval and time may be omitted. If eval is omitted it is assumed to be "unknown";
        if time is omitted it is assumed to be 0.

        Required:Update the game state by making the move. No response required.
        """
        # logger.debug(f"[{move}] [{evaluation}] [{time_sec}]")

        action = convert_move_to_action(move)
        self.engine.move(action)

    def hint(self, n):
        """Tell the engine to give evaluations for the given position. n tells how many moves to evaluate,
        e.g. 2 means give evaluations for the top 2 positions. This is used when the user is analyzing a game.
        With the "hint" command the engine is not CONSTRained by the time remaining in the game.

        Required: The engine sends back an evaluation for at its top move

        Best: The engine sends back an evaluation for approximately the top n moves.
        If the engine searches using iterative deepening it should also send back evaluations during search,
        which makes the GUI feel more responsive to the user.

        Depending on whether the evalation came from book or a search, the engine sends back

        search {pv: PV} {eval:Eval} 0 {depth:Depth} {freeform text}
        or
        book {pv: PV} {eval:Eval} {# games:long} {depth:Depth} {freeform text:string}

        PV: The pv must begin with two characters representing the move considered (e.g. "F5" or "PA") and
        must not contain any whitespace. "F5d6C3" and "F5-D6-C3" are valid PVs but "F5 D6 C3" will
        consider D6 to be the eval.

        Eval: The eval is from the point-of-view of the player to move and is a double.
        At the engine's option it can also be an ordered pair of doubles separated by a comma:
        {draw-to-black value}, {draw-to-white value}.

        Depth: depth is the search depth. It must start with an integer but can end with other characters;
        for instance "100%W" is a valid depth. The depth cannot contain spaces.

        Two depth codes have special meaning to NBoard: "100%W" tells NBoard that the engine has solved
        for a win/loss/draw and the sign of the eval matches the sign of the returned eval.
        "100%" tells NBoard that the engine has done an exact solve.
        The freeform text can be any other information that the engine wants to convey.
        NBoard 1.1 and 2.0 do not display this information but later versions or other GUIs may.

        :param n:
        """
        self.tell_status("thinkng hint...")
        self.engine.hint(int(n))
        self.tell_status("waiting")

    def report_hint(self, hint_list):
        for hint in reversed(hint_list):  # there is a rule that the last is best?
            move = convert_action_to_move(hint.action)
            self.engine.reply(f"search {move} {hint.value} 0 {int(hint.visit)}")

    def go(self):
        """Tell the engine to decide what move it would play.

        This is used when the engine is playing in a game.
        With the "go" command the computer is limited by both the maximum search depth and
        the time remaining in the game.

        Required: The engine responds with "=== {move}" where move is e.g. "F5"

        Best: The engine responds with "=== {move:String}/{eval:float}/{time:float}".
        Eval may be omitted if the move is forced. The engine also sends back thinking output
        as in the "hint" command.

        Important: The engine does not update the board with this move,
        instead it waits for a "move" command from NBoard.
        This is because the user may have modified the board while the engine was thinking.

        Note: To make it easier for the engine author,
        The NBoard gui sets the engine's status to "" when it receives the response.
        The engine can override this behaviour by sending a "status" command immediately after the response.
        """
        self.tell_status("thinking...")
        gr = self.engine.go()
        move = convert_action_to_move(gr.action)
        self.engine.reply(f"=== {move}/{gr.eval * 10}/{gr.time}")
        self.tell_status("waiting")

    def ping(self, n):
        """Ensure synchronization when the board position is about to change.

        Required: Stop thinking and respond with "pong n".
        If the engine is analyzing a position it must stop analyzing before sending "pong n"
        otherwise NBoard will think the analysis relates to the current position.
        :param n:
        :return:
        """
        # self.engine.stop_thinkng()  # not implemented
        self.engine.reply(f"pong {n}")

    def learn(self):
        """Learn the current game.
        Required: Respond "learned".

        Best: Add the current game to book.

        Note: To make it easier for the engine author,
        The NBoard gui sets the engine's status to "" when it receives the "learned" response.
        The engine can override this behaviour by sending a "status" command immediately after the response.
        """
        self.engine.reply("learned")

    def analyze(self):
        """Perform a retrograde analysis of the current game.

        Optional: Perform a retrograde analysis of the current game.
        For each board position occurring in the game,
        the engine sends back a line of the form analysis {movesMade:int} {eval:double}.
        movesMade = 0 corresponds to the start position. Passes count towards movesMade,
        so movesMade can go above 60.
        """
        pass

    def tell_status(self, status):
        self.engine.reply(f"status {status}")


