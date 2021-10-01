""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: azhou90 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 903741795 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		   	 		  		  		    	 		 		   		 		  
import os  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
def compute_portvals(  		  	   		   	 		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		   	 		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		   	 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		   	 		  		  		    	 		 		   		 		  
    impact=0.005
):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		   	 		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		   	 		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		  	   		   	 		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		   	 		  		  		    	 		 		   		 		  
    # code should work correctly with either input  		  	   		   	 		  		  		    	 		 		   		 		  
    # orders_df = pd.read_csv(orders_file, index_col="Date", parse_dates=True, na_values=["nan"]).sort_index()
    orders_df = pd.read_csv(orders_file, na_values=["nan"])
    # print(orders_df.columns)
    orders_df.loc[:,'Date'] = pd.to_datetime(orders_df.loc[:,'Date'])
    orders_df.sort_values(by='Date', inplace=True)
    # print(orders_df.head())

    cash = start_val
    port = 0
    start_date = orders_df.iloc[0,0]
    end_date = orders_df.iloc[-1,0]
    # print(start_date, end_date)
    total_range = pd.date_range(start=start_date, end=end_date)
    # print(total_range)
    symbols = list(orders_df.loc[:, 'Symbol'].unique())
    # print(symbols)
    sym_price = get_data(symbols, total_range)
    sym_price.ffill(inplace=True)
    sym_price.bfill(inplace=True)
    # print(sym_price.index)

    # print(sym_price.head())
    total_value = []
    # print(total_range[0])
    # print(orders_df.iloc[0,0])
    # print(total_range[0] == orders_df.iloc[0,0])
    # print(list(orders_df['Date']))
    # print(total_range[0] in list(orders_df.loc[:,'Date']))
    my_port = pd.DataFrame(columns=symbols, index=sym_price.index)
    my_port[:]=0
    # print(my_port)
    trade_dates = list(orders_df.loc[:,'Date'])
    for date in sym_price.index:
        # if date not in orders_df.loc[:,'Date']:
        #     value = cash + port
        #     total_value.append(value)
        if date in trade_dates:
            df_orders_today = orders_df.loc[orders_df['Date'] == date, :]
            # print(df_orders_today)
            for transac_idx in range(df_orders_today.shape[0]):
                transaction = df_orders_today.iloc[transac_idx, :]
                if transaction['Order'] == 'BUY':
                    # print(transaction)
                    share = transaction["Shares"]
                    sym = transaction["Symbol"]
                    price = sym_price.loc[date, sym]
                    my_port.loc[date:, sym] += share
                    # port = port + price * share
                    cash = cash - commission - price * share * (1 + impact)
                else:
                    share = transaction["Shares"]
                    sym = transaction["Symbol"]
                    price = sym_price.loc[date, sym]
                    my_port.loc[date:, sym] -= share
                    # port = port - price * share
                    cash = cash - commission + price * share * (1 - impact)

        port = my_port.loc[date] @ sym_price.loc[date , symbols]
        print(port)
        # print(my_port.loc[date] )
        # print(sym_price.loc[date , symbols])
        value = cash + port
        total_value.append(value)
    # print(my_port)
        # else:
        #     share = orders_df.loc[total_range[i]]["Shares"]
        #     sym = orders_df.loc[total_range[i]]["Symbol"]
        #     price = sym_price.loc[total_range[i],sym]
        #     # order = orders_df.loc[total_range[i]]["Order"]
        #     order = orders_df.loc[orders_df.index == total_range[i]]
        #     print(order)
        #     if order == "BUY":
        #         port = port + price * share * (1+impact)
        #         cash = cash - commission - price * share * (1+impact)
        #         value = cash + port
        #         total_value.append(value)
        #     else:
        #         port = port + price * share * (1 - impact)
        #         cash = cash - commission - price * share * (1 - impact)
        #         value = cash + port
        #         total_value.append(value)
    df_out = pd.DataFrame(data=total_value, index=sym_price.index, columns=["equity"])
    print(df_out)
    return df_out
  		  	   		   	 		  		  		    	 		 		   		 		  
    # In the template, instead of computing the value of the portfolio, we just  		  	   		   	 		  		  		    	 		 		   		 		  
    # read in the value of IBM over 6 months  		  	   		   	 		  		  		    	 		 		   		 		  
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # portvals = get_data(["IBM"], pd.date_range(start_date, end_date))
    # portvals = portvals[["IBM"]]  # remove SPY
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)
  		  	   		   	 		  		  		    	 		 		   		 		  
    # return rv
    # return portvals

def author():
    return "azhou90"
  		  	   		   	 		  		  		    	 		 		   		 		  
def test_code():  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		   	 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		   	 		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    of = "./orders/orders2.csv"  		  	   		   	 		  		  		    	 		 		   		 		  
    sv = 1000000  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # Process orders  		  	   		   	 		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		   	 		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		  	   		   	 		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		   	 		  		  		    	 		 		   		 		  
    else:  		  	   		   	 		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # Get portfolio stats  		  	   		   	 		  		  		    	 		 		   		 		  
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		   	 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008, 1, 1)  		  	   		   	 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2008, 6, 1)  		  	   		   	 		  		  		    	 		 		   		 		  
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [  		  	   		   	 		  		  		    	 		 		   		 		  
        0.2,  		  	   		   	 		  		  		    	 		 		   		 		  
        0.01,  		  	   		   	 		  		  		    	 		 		   		 		  
        0.02,  		  	   		   	 		  		  		    	 		 		   		 		  
        1.5,  		  	   		   	 		  		  		    	 		 		   		 		  
    ]  		  	   		   	 		  		  		    	 		 		   		 		  
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		   	 		  		  		    	 		 		   		 		  
        0.2,  		  	   		   	 		  		  		    	 		 		   		 		  
        0.01,  		  	   		   	 		  		  		    	 		 		   		 		  
        0.02,  		  	   		   	 		  		  		    	 		 		   		 		  
        1.5,  		  	   		   	 		  		  		    	 		 		   		 		  
    ]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":
    compute_portvals(
        orders_file="./orders/orders-05.csv",
        start_val=1000000,
        commission=0.5,
        impact=0.005
    )
