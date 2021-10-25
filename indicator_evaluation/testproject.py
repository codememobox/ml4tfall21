# This file should be considered the entry point to the project.
# The if “__name__” == “__main__”: section of the code will call the testPolicy function in
# TheoreticallyOptimalStrategy, as well as your indicators and marketsimcode as needed,
# to generate the plots and statistics for your report (more details below).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt

from util import get_data
import TheoreticallyOptimalStrategy as tos

def author():
    return "azhou90"

df_trades = tos.testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)