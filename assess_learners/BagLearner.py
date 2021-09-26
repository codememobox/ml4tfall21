
import numpy as np


class BagLearner(object):
    def __init__(self, learner=None, kwargs={}, bags=20,
                 boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = [self.learner(**kwargs) for i in range(self.bags)]

    def author(self):
        return "azhou90"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        def bag_data(data_x, data_y):
            data_select = np.random.randint(0,data_y.shape[0], data_y.shape[0])
            return data_x[data_select], data_y[data_select]


        for learner in self.learners:
            select_x, select_y = bag_data(data_x, data_y)
            learner.add_evidence(select_x, select_y)

    def query(self, points):
        predict = np.zeros(points.shape[0])
        for learner in self.learners:
            predict= predict + learner.query(points)
        predict= predict/len(self.learners)
        return predict


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
