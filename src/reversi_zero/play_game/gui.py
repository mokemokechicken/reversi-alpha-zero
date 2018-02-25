# many code from http://d.hatena.ne.jp/yatt/20100129/1264791420

from logging import getLogger

import wx
from wx.core import CommandEvent

from reversi_zero.config import Config, GuiConfig, PlayWithHumanConfig
from reversi_zero.env.reversi_env import Player
from reversi_zero.play_game.game_model import PlayWithHuman, GameEvent

logger = getLogger(__name__)


def start(config: Config):
    config.play_with_human.update_play_config(config.play)
    reversi_model = PlayWithHuman(config)
    app = wx.App()
    Frame(reversi_model, config.gui).Show()
    app.MainLoop()


def notify(caption, message):
    dialog = wx.MessageDialog(None, message=message, caption=caption, style=wx.OK)
    dialog.ShowModal()
    dialog.Destroy()


class Frame(wx.Frame):
    def __init__(self, model: PlayWithHuman, gui_config: GuiConfig):
        self.model = model
        self.gui_config = gui_config
        self.is_flip_vertical = False
        self.show_player_evaluation = True
        wx.Frame.__init__(self, None, -1, self.gui_config.window_title, size=self.gui_config.window_size)
        # panel
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self.try_move)
        self.panel.Bind(wx.EVT_PAINT, self.refresh)

        self.new_game(human_is_black=True)
        # menu bar
        menu = wx.Menu()
        menu.Append(1, u"New Game(Black)")
        menu.Append(2, u"New Game(White)")
        menu.AppendSeparator()
        menu.Append(5, u"Flip Vertical")
        menu.Append(6, u"Show/Hide Player evaluation")
        menu.AppendSeparator()
        menu.Append(9, u"quit")
        menu_bar = wx.MenuBar()
        menu_bar.Append(menu, u"menu")
        self.SetMenuBar(menu_bar)
        self.Bind(wx.EVT_MENU, self.handle_new_game, id=1)
        self.Bind(wx.EVT_MENU, self.handle_new_game, id=2)
        self.Bind(wx.EVT_MENU, self.handle_flip_vertical, id=5)
        self.Bind(wx.EVT_MENU, self.handle_show_hide_player_evaluation, id=6)
        self.Bind(wx.EVT_MENU, self.handle_quit, id=9)

        # status bar
        self.CreateStatusBar()

        self.model.add_observer(self.handle_game_event)

    def handle_game_event(self, event):
        if event == GameEvent.update:
            self.panel.Refresh()
            self.update_status_bar()
            wx.Yield()
        elif event == GameEvent.over:
            self.game_over()
        elif event == GameEvent.ai_move:
            self.ai_move()

    def handle_quit(self, event: CommandEvent):
        self.Close()

    def handle_new_game(self, event: CommandEvent):
        self.new_game(human_is_black=event.GetId() == 1)

    def handle_flip_vertical(self, event):
        self.is_flip_vertical = not self.is_flip_vertical
        self.panel.Refresh()

    def handle_show_hide_player_evaluation(self, event):
        self.show_player_evaluation = not self.show_player_evaluation
        self.panel.Refresh()

    def new_game(self, human_is_black):
        self.model.start_game(human_is_black=human_is_black)
        self.model.play_next_turn()

    def ai_move(self):
        self.panel.Refresh()
        self.update_status_bar()
        wx.Yield()
        self.model.move_by_ai()
        self.model.play_next_turn()

    def try_move(self, event):
        if self.model.over:
            return
        # calculate coordinate from window coordinate
        event_x, event_y = event.GetX(), event.GetY()
        w, h = self.panel.GetSize()
        x = int(event_x / (w / 8))
        y = int(event_y / (h / 8))

        if self.is_flip_vertical:
            y = 7-y

        if not self.model.available(x, y):
            return

        self.model.move(x, y)
        self.model.play_next_turn()

    def game_over(self):
        # if game is over then display dialog

        black, white = self.model.number_of_black_and_white
        mes = "black: %d\nwhite: %d\n" % (black, white)
        if black == white:
            mes += "** draw **"
        else:
            mes += "winner: %s" % ["black", "white"][black < white]
        notify("game is over", mes)
        # elif self.reversi.passed != None:
        #     notify("passing turn", "pass")

    def update_status_bar(self):
        msg = "current player is " + ["White", "Black"][self.model.next_player == Player.black]
        if self.model.last_evaluation:
            msg += f"|AI Confidence={self.model.last_evaluation:.4f}"
        self.SetStatusText(msg)

    def refresh(self, event):
        dc = wx.PaintDC(self.panel)
        self.update_status_bar()

        w, h = self.panel.GetSize()
        # background
        dc.SetBrush(wx.Brush("#228b22"))
        dc.DrawRectangle(0, 0, w, h)
        # grid
        dc.SetBrush(wx.Brush("black"))
        px, py = w / 8, h / 8
        for y in range(8):
            dc.DrawLine(y * px, 0, y * px, h)
            dc.DrawLine(0, y * py, w, y * py)
        dc.DrawLine(w - 1, 0, w - 1, h - 1)
        dc.DrawLine(0, h - 1, w - 1, h - 1)

        # stones
        brushes = {Player.white: wx.Brush("white"), Player.black: wx.Brush("black")}
        for y in range(8):
            vy = 7-y if self.is_flip_vertical else y
            for x in range(8):
                c = self.model.stone(x, y)
                if c is not None:
                    dc.SetBrush(brushes[c])
                    dc.DrawEllipse(x * px, vy * py, px, py)
                if self.model.last_history:
                    q_value = self.model.last_history.values[y*8+x]
                    n_value = self.model.last_history.visit[y*8+x]
                    enemy_q_value = - self.model.last_history.enemy_values[y*8+x]
                    enemy_n_value = self.model.last_history.enemy_visit[y*8+x]

                    dc.SetTextForeground(wx.Colour("blue"))
                    if n_value:
                        dc.DrawText(f"{int(n_value):d}", x*px+2, vy*py+2)
                    if q_value:
                        if q_value < 0:
                            dc.SetTextForeground(wx.Colour("red"))
                        dc.DrawText(f"{int(q_value*100):d}", x*px+2, (vy+1)*py-16)

                    if self.show_player_evaluation:
                        dc.SetTextForeground(wx.Colour("purple"))
                        if enemy_n_value:
                            dc.DrawText(f"{int(enemy_n_value):2d}", (x+1)*px-20, vy*py+2)
                        if enemy_q_value:
                            if enemy_q_value < 0:
                                dc.SetTextForeground(wx.Colour("orange"))
                            dc.DrawText(f"{int(enemy_q_value*100):2d}", (x+1)*px-24, (vy+1)*py-16)
