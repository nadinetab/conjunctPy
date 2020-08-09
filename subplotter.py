"""
Nadine Tabucol

Quick(er) Matplotlib multiple plotting, esp. for time-series variables

"""

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from datetime import datetime
import matplotlib.dates as mdates
import pandas as pd

def create_subplots(n, x_vars, y_vars, set_ylimits = None,
                    suptitles = None, plot_title = None, plotsize = (10,15), plot_datetime = False):
    """
        Creates n vertically-stacked subplots.

        Kwargs:
            set_ylimits : list of tuples
                List of tuples containing y-axis limits for each variable to plot.
                Format as (bottom limit, top limit) for each variable.
                Default 'None'
            suptitles : list
                List of subplot titles.
                Default 'None'
            plot_title : str
                Title for entire figure. Default 'None'.
            plotsize : tuple
                Size of all subplots (combined). See matplotlib.pyplot.figure.
                Default (10, 15)
            plot_datetime : bool
                'True' if x-axis var. is of datetime (from datetime or NumPy or Pandas) type.
                Default 'False' 
        Args:
            n : int
                Number of subplots.
            x_vars : list, size n
                List of x-axis variables, order corresponding to that of y_vars.
            y_vars : list, size n
                List of y-axis variables, order corresponding to that of x_vars.

    """

    fig, axs = plt.subplots(nrows = n,figsize = plotsize)

    if plot_datetime:
        register_matplotlib_converters()
        fig.autofmt_xdate() # Rotate and align tick labels

    if plot_title:
        fig.suptitle(plot_title)

    for i in range(n):

        if plot_datetime:
            timeX = pd.to_datetime(x_vars[i], format='%Y-%m-%d %H:%M:%S')

            firstX = timeX.iloc[0]
            lastX = timeX.iloc[-1]
            
            if firstX.date() == lastX.date():
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            else:
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

            if suptitles:
                axs[i].set_title(suptitles[i] + ' -- ' + firstX.strftime('%Y-%m-%d'))

            axs[i].plot(timeX, y_vars[i], '-o')

        else:
            axs[i].plot(x_vars[i], y_vars[i], '-o')
            
            if suptitles:
                axs[i].set_title(suptitles[i])

        if set_ylimits:
            axs[i].set_ylim(bottom=set_ylimits[i][0], top=set_ylimits[i][1])
