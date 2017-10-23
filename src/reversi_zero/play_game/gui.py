# many code from http://d.hatena.ne.jp/yatt/20100129/1264791420

from logging import getLogger

import wx
from wx.core import CommandEvent

from reversi_zero.config import Config, GuiConfig
from reversi_zero.env.reversi_env import Player
from reversi_zero.play_game.game_model import PlayWithHuman

logger = getLogger(__name__)


def start(config: Config):
    reversi_model = PlayWithHuman(config)
    app = wx.PySimpleApp()
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
        wx.Frame.__init__(self, None, -1, self.gui_config.window_title, size=self.gui_config.window_size)
        # panel
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self.try_move)
        self.panel.Bind(wx.EVT_PAINT, self.refresh)

        # self.new_game(None)
        # menu bar
        menu = wx.Menu()
        menu.Append(1, u"New Game(Black)")
        menu.Append(2, u"New Game(White)")
        menu.AppendSeparator()
        menu.Append(9, u"quit")
        menu_bar = wx.MenuBar()
        menu_bar.Append(menu, u"menu")
        self.SetMenuBar(menu_bar)
        self.Bind(wx.EVT_MENU, self.new_game, id=1)
        self.Bind(wx.EVT_MENU, self.new_game, id=2)
        self.Bind(wx.EVT_MENU, self.quit, id=9)

        # status bar
        self.CreateStatusBar()

        self.model.add_observer(self.handle_game_event)

    def handle_game_event(self, event):
        pass

    def quit(self, event: CommandEvent):
        self.Close()

    def new_game(self, event: CommandEvent):
        event.GetId()
        # initialize reversi and refresh screen
        logger.debug(f"event id={event.GetId()}")
        human_is_black = event.GetId() == 1
        self.model.start_game(human_is_black=human_is_black)
        self.panel.Refresh()

    def try_move(self, event):
        if self.model.over:
            return
        # calculate coordinate from window coordinate
        event_x, event_y = event.GetX(), event.GetY()
        w, h = self.panel.GetSize()
        x = event_x / (w / 8)
        y = event_y / (h / 8)
        if not self.model.available(x, y):
            return

        self.model.move(x, y)
        self.panel.Refresh()

        # # if game is over then display dialog
        # if self.reversi.over:
        #     black = self.reversi.stones(BLACK)
        #     white = self.reversi.stones(WHITE)
        #     mes = "black: %d\nwhite: %d\n" % (black, white)
        #     if black == white:
        #         mes += "** draw **"
        #     else:
        #         mes += "winner: %s" % ["black", "white"][black < white]
        #     notify("game is over", mes)
        # elif self.reversi.passed != None:
        #     notify("passing turn", "pass")

    def refresh(self, event):
        dc = wx.PaintDC(self.panel)
        self.SetStatusText("current player is " + ["White", "Black"][self.model.next_player == Player.black])
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
            for x in range(8):
                c = self.model.stone(x, y)
                if c is not None:
                    dc.SetBrush(brushes[c])
                    dc.DrawEllipse(y * px, x * py, px, py)
        #         elif self.reversi.available(i, j):
        #             dc.SetBrush(wx.Brush("red"))
        #             dc.DrawCircle(i * px + px / 2, j * py + py / 2, 3)

