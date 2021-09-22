""""""
import matplotlib.pyplot as plt

"""Assess a betting strategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it availabel on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Anlu Zhou 		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: azhou90		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 903741795 		  	   		   	 		  		  		    	 		 		   		 		  
"""

import numpy as np
import matplotlib.pyplot as plt


def author():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    return "azhou90"  # replace tb34 with your Georgia Tech username.


def gtid():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    return 903741795  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		   	 		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def experiment(win_prob,real):
    bank = real * 256
    winnings = np.zeros(1000)
    
    n = 0
    bet_amount = 1

    while bank < (real*256)+80 and n < 1000:
        if real:
            if bank <= 0:
                return winnings
        won = get_spin_result(win_prob)
        if won:
            bank += bet_amount
            bet_amount = 1
        else:
            bank -= bet_amount
            bet_amount *= 2
            if real:
                if bet_amount > bank:
                    bet_amount = bank

        winnings[n:] = bank-(real*256)
        n += 1
    return winnings

def figure_1(win_prob, simulations):
    for i in range(simulations):
        plt.plot(experiment(win_prob, False))

    axes = plt.gca()
    axes.set_xlim([0,300])
    axes.set_ylim([-256,100])
    plt.title("figure_1: 10 simulations on experiment_1")
    plt.xlabel("bets")
    plt.ylabel("winnings")
    plt.legend(["simulation1","simulation2","simulation3","simulation4","simulation5","simulation6","simulation7","simulation8","simulation9","simulation10"])
    plt.savefig("figure_1")
    plt.clf()

def figure_2_3(win_prob,simulations):
    simulate_winning = []
    for i in range(simulations):
        simulate_winning.append(experiment(win_prob, False))

    simulate_winning = np.array(simulate_winning)
    simulate_mean = np.mean(simulate_winning, axis=0)
    simulate_std = np.std(simulate_winning, axis=0)

    # expected winning
    print('expected winning:', simulate_mean[-1])
    # probability of win
    print('prob of winning 80:',
          np.sum(simulate_winning[:,-1] == 80) / simulations)


    plt.plot(simulate_mean)
    plt.plot(simulate_mean + simulate_std)
    plt.plot(simulate_mean - simulate_std)
    axes = plt.gca()
    axes.set_xlim([0,300])
    axes.set_ylim([-256,100])
    plt.title("figure_2: 1000 simulations on experiment_1 with mean")
    plt.xlabel("bets")
    plt.ylabel("winings")
    plt.legend(["mean","mean-std","mean+std"])
    plt.savefig("figure_2")
    plt.clf()

    simulate_median = np.median(simulate_winning, axis=0)
    plt.plot(simulate_median)
    plt.plot(simulate_median + simulate_std)
    plt.plot(simulate_median - simulate_std)
    axes = plt.gca()
    axes.set_xlim([0, 300])
    axes.set_ylim([-256, 100])
    plt.title("figure_3: 1000 simulations on experiment_1 with median")
    plt.xlabel("bets")
    plt.ylabel("winings")
    plt.legend(["median","median-std","median+std"])
    plt.savefig("figure_3")
    plt.clf()

def figure_4_5(win_prob, simulations):
    simulate_winning = []
    for i in range(simulations):
        simulate_winning.append(experiment(win_prob,True))

    simulate_winning = np.array(simulate_winning)
    simulate_mean = np.mean(simulate_winning, axis=0)
    simulate_std = np.std(simulate_winning, axis=0)

    # expected winning
    print('expected winning:', simulate_mean[-1])
    # probability of win
    print('prob of winning 80:',
          np.sum(simulate_winning[:, -1] == 80) / simulations)

    plt.plot(simulate_mean)
    plt.plot(simulate_mean + simulate_std)
    plt.plot(simulate_mean - simulate_std)
    axes = plt.gca()
    axes.set_xlim([0, 300])
    axes.set_ylim([-256, 100])
    plt.title("figure_4: 1000 simulations on experiment_2 with mean")
    plt.xlabel("bets")
    plt.ylabel("winings")
    plt.legend(["mean", "mean-std", "mean+std"])
    plt.savefig("figure_4")
    plt.clf()

    simulate_median = np.median(simulate_winning, axis=0)
    plt.plot(simulate_median)
    plt.plot(simulate_median + simulate_std)
    plt.plot(simulate_median - simulate_std)
    axes = plt.gca()
    axes.set_xlim([0, 300])
    axes.set_ylim([-256, 100])
    plt.title("figure_5: 1000 simulations on experiment_2 with median")
    plt.xlabel("bets")
    plt.ylabel("winings")
    plt.legend(["median", "median-std", "median+std"])
    plt.savefig("figure_5")
    plt.clf()


def test_code():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    win_prob = 18/38 # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once  		  	   		   	 		  		  		    	 		 		   		 		  
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		   	 		  		  		    	 		 		   		 		  
    # add your code here to implement the experiments  		  	   		   	 		  		  		    	 		 		   		 		  
    figure_1(win_prob,10)
    figure_2_3(win_prob, 1000)
    figure_4_5(win_prob, 1000)

if __name__ == "__main__":
    test_code()
