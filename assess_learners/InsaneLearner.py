import numpy as np
import LinRegLearner as lrl
import BagLearner as bl


class InsaneLearner(object):

    def __init__(self, verbose=False):
        self.learner = bl.BagLearner(bl.BagLearner,
                                     kwargs={'learner':lrl.LinRegLearner, 'bags':20},
                                     bags=20)

    def author(self):
        return "azhou90"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        self.learner.add_evidence(data_x,data_y)

    def query(self, points):
        return self.learner.query(points)

