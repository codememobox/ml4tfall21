
import numpy as np

class DTLearner(object):


    def __init__(self, verbose=False, leaf_size = 1):
        """
        Constructor method
        """
        self.verbose = verbose
        self.leaf_size = leaf_size
        self.tree = None

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "azhou90"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        '''
        train a decision tree and add it to the constructor
        ---
        INPUT
        data_x : features
        data_y : labels
        '''

        def build_tree(data_x, data_y):
            '''
            Helper function: Train a decision tree from labeled data
            ---
            INPUT
            data_x : features
            data_y : labels
            ---
            OUTPUT
            tree : numpy matrix
            ---
            NOTE
            each row/node : [feature, split_val, left_to_go, right_to_go]
            leaf node has an index of -1
            '''

            # base case: if the tree is not divisible
            if data_y.shape[0] <= self.leaf_size or np.var(data_y) == 0:
                # return the leaf node
                return np.array([[-1, np.mean(data_y), None, None]])
            else:
                # determine the best feature by correlation
                corr = np.abs(np.corrcoef(data_x, data_y, rowvar=False))[:-1, -1]
                # i denotes the feature w/ highest abs correlation
                i = np.argmax(corr)
                # median of the chosen feature is the split threshold
                SplitVal = np.median(data_x[:, i])

                # under threshold: throw data entries to the left tree
                left_mask = data_x[:, i] <= SplitVal
                # above threshold: throw data entries to the right tree
                right_mask = data_x[:, i] > SplitVal

                # if there's nothing to split
                if np.all(left_mask) or np.all(right_mask):
                    # return leaf node
                    return np.array([[-1,np.mean(data_y), None, None]])

                # recursion step: further split left tree
                lefttree = build_tree(
                    data_x[left_mask, :],
                    data_y[left_mask]
                )
                # recursion step: further split left tree
                righttree = build_tree(
                    data_x[right_mask, :],
                    data_y[right_mask]
                )

                # create tree entry for current node
                node = np.array([[i, SplitVal, 1, lefttree.shape[0] + 1]])
                # grow tree recursively
                tree = np.vstack((node, lefttree, righttree))
                return tree

        self.tree = build_tree(data_x, data_y)
        # print(self.tree)

    def query(self, data_x):
        """
        Predict labels of data using the trained decision tree
        ---
        INPUT
        data_x : data features
        ---
        OUTPUT
        predicted labels : a list of predicted labels
        """
        predicted_labels = []
        for x in data_x:
            # start from the root node
            row = 0
            # retrieve the node information
            node = self.tree[row, :]
            # keep proceeding if node is not leaf
            while node[0] != -1:
                # pick up the feature
                feature_idx = int(node[0])
                # if feature < threshold: proceed to the left tree
                if x[feature_idx] <= node[1]:
                    row += int(node[2])
                    node = self.tree[row, :]
                # if feature > threshold: proceed to the right tree
                else:
                    row += int(node[3])
                    node = self.tree[row, :]
            predicted_labels.append(node[1])
        return predicted_labels


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")