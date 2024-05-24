from enum import Enum

# https://www.pysimplegui.org/en/latest/
import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from configuration import ROOT_PATH
from app.model import Model

# setup constants
GUI_WINDOW_TITLE = 'Data Science Template'
THEME = 'DarkGrey11'
sg.theme(THEME)
TEXT_BG_COLOR = sg.theme_background_color()
DEFAULT_FONT = ('Helvetica', 15)

layout = [
    [
        sg.Text('Hello', background_color=TEXT_BG_COLOR)
    ],
    [
        sg.Text('Last Entered Text:', background_color=TEXT_BG_COLOR),
        sg.Text(size=(15, 1), key='-OUTPUT-', background_color=TEXT_BG_COLOR)
    ],
    [
        sg.Text('Enter some text', background_color=TEXT_BG_COLOR), sg.Input(key='-IN-')
    ],
    [
        sg.Button('Model')
    ],
    [
        sg.Canvas(key='-CANVAS-')
    ],
    [
        sg.Button('Ok'), sg.Exit()
    ]
]


class ExitCode(Enum):
    EXIT = 0
    OK = 1


class MainWindow:
    def __init__(self) -> None:
        self.window = sg.Window(GUI_WINDOW_TITLE, layout, font=DEFAULT_FONT, finalize=True)

        self.test_path = ROOT_PATH.joinpath("data/data.csv")

    def configure(self) -> None:
        self.model = Model()
        self.model.load_data(self.test_path)
        sns.set_theme(style="darkgrid")

        # include matplotlib figure in the tkinter window
        fig = plt.gcf()
        self.fig_agg = FigureCanvasTkAgg(fig, self.window['-CANVAS-'].TKCanvas)
        self.fig_agg.draw()
        self.fig_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

    def _plot_trained_model(self) -> None:
        fig = plt.gcf()
        ax = fig.gca()
        ax.clear()

        # plot the model data
        data = self.model.X.join(self.model.y)
        sns.scatterplot(data=data, x='x', y='y')

        # plot the model
        model_params = self.model.get_model_parameters()
        x = np.array(ax.get_xlim()).reshape(-1, 1)
        y = model_params.slope * x + model_params.intercept
        plt.plot(x, y, '-', color='red')
        self.fig_agg.draw()

    def loop(self, debug: bool = False) -> ExitCode:
        event, values = self.window.read()
        if debug:
            print("=" * 50)
            print(event)
            print(values)
        if event in (sg.WINDOW_CLOSED, 'Exit'):
            return ExitCode.EXIT
        if event == 'Ok':
            self.window['-OUTPUT-'].update(values['-IN-'])
        elif event == 'Model':
            self.model.load_data(self.test_path)  # generate new random data for the example
            self.window.perform_long_operation(self.model.train, '-TRAINING DONE-')
        elif event == '-TRAINING DONE-':
            self._plot_trained_model()
        return ExitCode.OK

    def close(self) -> None:
        self.window.close()


def main() -> None:
    matplotlib.use('TkAgg')
    main_window = MainWindow()
    main_window.configure()
    while True:
        exit_code = main_window.loop()
        if exit_code == ExitCode.EXIT:
            break
    main_window.close()


if __name__ == '__main__':
    main()
