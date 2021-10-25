# Code implementing a TheoreticallyOptimalStrategy (details below).
# It should implement testPolicy(), which returns a trades data frame (see below).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()
import os
import datetime as dt
import marketsimcode as ms

from util import get_data

def author():
    return "azhou90"

def get_price(sym, start_date, end_date):
    '''
    Helper function: get prices given symbol
    :param sym:
    :param start_date:
    :param end_date:
    :return:
    '''
    if not isinstance(sym, list):
        sym = [sym]
    dates = pd.date_range(start=start_date, end=end_date)
    prices = get_data(sym, dates, addSPY=False)
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    # Normalize price
    # prices = prices/prices.iloc[0]
    return prices

def benchmarkPolicy(symbol, sd, ed, sv):
    def benchmarkOrder(symbol, sd, ed):
        '''
        Helper function: get benchmark order
        :param symbol:
        :param sd:
        :param ed:
        :return:
        '''
        prices = get_price(symbol, sd, ed)
        orders_df = pd.DataFrame({'Date': [prices.index[1], prices.index[-1]],
                                  'Symbol': [symbol,symbol],
                                  'Order': ['BUY', 'BUY'],
                                  'Shares': [1000,0]})

        return orders_df

    benchmark_orders_df = benchmarkOrder(symbol, sd, ed)
    # print(benchmark_orders_df)
    benchmark_val = ms.compute_portvals(benchmark_orders_df, start_val=sv, commission=0, impact=0)
    return benchmark_val

def testPolicy(symbol, sd, ed, sv):
    def testOrder(symbol, sd, ed):
        '''
        Helper function: get test policy order
        :param symbol:
        :param sd:
        :param ed:
        :return:
        '''
        orders_df = get_price(symbol, sd, ed)
        orders_df['Symbol'] = symbol
        orders_df.rename(columns={symbol: 'Price'}, inplace=True)
        orders_df['Order'] = np.where(orders_df['Price'] < orders_df['Price'].shift(-1), 'BUY', 'SELL')
        actions = 1000 * np.where(orders_df['Price'] < orders_df['Price'].shift(-1), 1, -1)
        # print(actions)
        shares = np.zeros(len(actions)+1)
        for day in range(1,len(shares)):
            shares[day] = shares[day-1] + actions[day-1]
            if shares[day] > 1000:
                actions[day-1] = 0
                shares[day] = 1000
            if shares[day] < -1000:
                actions[day-1] = 0
                shares[day] = -1000
        # print(shares)
        # print(actions)
        orders_df['Shares'] = np.abs(actions)
        # orders_df =orders_df.iloc[actions!=0,:]
        orders_df = orders_df.iloc[:,1:]
        orders_df.reset_index(inplace=True)
        orders_df.rename(columns={'index': 'Date'}, inplace=True)
        # orders_df.iloc[1:,1:] = orders_df.iloc[:,1:].shift(1)
        # print(orders_df)
        # orders_df = orders_df.iloc[1:,:]
        # print(orders_df.iloc[:-1,:])
        return orders_df

    test_orders_df = testOrder(symbol, sd, ed)
    # print(test_orders_df)
    test_val = ms.compute_portvals(test_orders_df, start_val=sv, commission=0, impact=0)

    return test_val




if __name__ == "__main__":
    sym = 'JPM'
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009,12,31)
    sv = 100000

    bench = benchmarkPolicy(symbol=sym, sd = start_date, ed = end_date, sv=sv)
    bench = bench / bench.iloc[0]
    # print(bench)

    test = testPolicy(symbol=sym, sd = start_date, ed = end_date, sv = sv)
    test = test/test.iloc[0]
    # print(test)

    # plt.figure(figsize=(10,5))
    # plt.plot(bench, c='g', label='Benchmark')
    # plt.plot(test, c='r', label='Test')
    # plt.title('Theoretically Optimal Strategy')
    # plt.xlabel('Date')
    # plt.ylabel('Normalized portfolio value')
    # plt.legend()
    # plt.savefig('./fig/tos.png')
    # plt.show()