import numpy as np
from imblearn.over_sampling import SMOTE

'''
complete with SMOTE-WENN_Solving_class_imbalance_and_small_sampl.pdf
'''

def distance_a(x, y, std):
    return abs(x - y) / 4 * std

def distance_hvdm(x_1, x_2, std):
    # only continus
    distance = .0
    for j in range(0, x_1.shape[0]):
        distance += distance_a(x_1[j], x_2[j], std[j]) ** 2
    return distance ** .5



class SMOTEWENN:
    def __init__(self, p = 5, k = 5, m = 2, minority_class=None, n=None, random_state = 1):
        self.p = p
        self.k = k
        self.m = m
        self.minority_class = minority_class
        self.n = n
        self.random_state = random_state

    def fit_sample(self, X, y):
        if self.minority_class is None:
            classes = np.unique(y)
            sizes = [sum(y == c) for c in classes]
            minority_class = classes[np.argmin(sizes)]
        else:
            minority_class = self.minority_class

        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()

        if self.n is None:
            n = len(majority_points) - len(minority_points)  # 差异
        else:
            n = self.n

        IR_positive = float(len(minority_points)) / float(len(majority_points) + len(minority_points))
        IR_negative = float(len(majority_points)) / float(len(majority_points) + len(minority_points))

        # do smote
        smo = SMOTE(random_state=self.random_state)
        X_smo, y_smo = smo.fit_sample(X, y)  # get New_Tr

        distance_matrix = np.zeros((X_smo.shape[0], X_smo.shape[0]))

        # compute attribute sd
        '''
        minor_sd = np.zeros(minority_points.shape[1])
        major_sd = np.zeros(majority_points.shape[1])
        for j in range(0, minority_points.shape[1]):
            data = minority_points[:,j]
            minor_sd[j] = np.std(data) # todo no seperated in paper

        for j in range(0, majority_points.shape[1]):
            data = majority_points[:,j]
            major_sd[j] = np.std(data)
        z = 1
        '''
        X_smo_std = np.zeros(X_smo.shape[1])
        for j in range(0, X_smo.shape[1]):
            data = X_smo[:,j]
            X_smo_std[j] = np.std(data)


        for i in range(0, X_smo.shape[0] ):
            for j in range(0, X_smo.shape[0]):
                if i == j: continue
                # judge x2
                IR = IR_negative
                if y_smo[j] == 1:
                    IR = IR_positive
                distance = distance_hvdm(X_smo[i], X_smo[j], X_smo_std)
                distance *= np.exp(IR ** self.m)
                distance_matrix[i, j] = distance

        # Remove Noisy
        Xpreserved = []
        Ypreserved = []
        for i in range(0, X_smo.shape[0]):
            distanceList = distance_matrix[i] # i th sample
            distanceSort = np.argsort(distanceList)
            nearstSample = distanceSort[1:self.k + 1]
            label = y_smo[i]
            minL = 0
            marL = 0
            for j in nearstSample:
                if y_smo[j] == 1:
                    minL += 1
                else:
                    marL += 1
            predict = 0
            if minL > marL:
                predict = 1
            if predict == label:
                Xpreserved.append(X_smo[i])
                Ypreserved.append(y_smo[i])
            z = 1
        return Xpreserved, Ypreserved






