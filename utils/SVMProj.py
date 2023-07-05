import numpy as np

# MaxMin-L2-SVC-NCH
class SVMProj:
    def __init__(self, X, y):
        # Initialize alphas
        m = len(y)
        alphas = np.ones(m).reshape(m, 1)
        pos_alpha = len(y[y == 1])
        neg_alpha = len(y[y == -1])
        for i in range(len(y)):
            if (y[i] == 1):
                alphas[i] = alphas[i] * (1 / pos_alpha)
            else:
                alphas[i] = alphas[i] * (1 / neg_alpha)
        y_pos = y.copy()
        y_pos[y_pos == -1] = 0
        y_neg = y.copy()
        y_neg[y_neg == 1] = 0
        y_neg = -y_neg

        self.X = X  # Training samples
        self.y = y  # Training labels
        self.y_pos = y_pos.reshape(len(y_pos), 1)  # The mask of positive samples
        self.y_neg = y_neg.reshape(len(y_neg), 1)  # The mask of the negative sample
        self.delt = 1  # Penalty parameter, 1/C
        self.gamma = 0.004  # Gaussian kernel parameter, 1/(2*sigma^2)
        self.alphas = alphas  # lagrange multiplier
        self.init_alphas = alphas
        self.threshold = 0.000001  # alphas threshold
        self.stopaccuracy = 0.001  # Accuracy of gamma loss
        self.lossIter = 20  # The number of iterations of the loss function
        self.gammaIter=500  # The maximum number of iterations of gamma
        self.max_epoch = 2000  # The maximum number of iterations for training
        self.m = m  # Number of training samples
        self.norms = [[0] * self.m for _ in range(self.m)]  # Norm matrix, ||xi-xj||^2
        self.gauss = [[0] * self.m for _ in range(self.m)]  # Gaussian kernel matrix, e^(-gamma*||xi-xj||^2)
        self.matrix = [[0] * self.m for _ in range(self.m)]  # G(alpha), yi*yj*e^(-gamma*||xi-xj||^2)
        self.d = np.zeros(self.m)  # alphas gradient, (yi*yj*e^(-gamma*||xi-xj||^2)+delt)*alpha
        self.stop = False  # Determine whether to stop training

    # Decision function for classification
    def decision_function(self, X):
        # Calculate the bias b of the decision function
        s = np.sum(self.alphas.reshape(1, len(self.y)) * self.y.reshape(1, len(self.y)) * self.gauss, axis=1).reshape(1,
                                                                                                                      len(self.alphas)) + self.alphas.reshape(
            1, len(self.alphas)) * self.y.reshape(1, len(self.y)) * self.delt

        a = self.alphas.copy()
        a[a <= self.threshold] = 0
        a[a != 0] = 1

        y_pos = self.y_pos.reshape(len(self.y_pos), 1) * a
        p = (y_pos.reshape(1, len(y_pos)) * s).sum() / len(y_pos[y_pos == 1])

        y_neg = self.y_neg.reshape(len(self.y_neg), 1) * a
        q = (y_neg.reshape(1, len(y_pos)) * s).sum() / len(y_neg[y_neg == 1])
        b = -(p + q) / 2

        y = np.array(self.y).reshape(len(self.y), 1)
        a = np.array(self.alphas)[:np.newaxis]
        flag = y * a
        ya = [flag for _ in range(X.shape[0])]
        if (X.shape[0] == self.m):
            # To quickly calculate the training set accuracy, use the pre-stored Gaussian matrix of the training set
            gus = self.gauss
        else:
            norms_test = [[0] * self.m for _ in range(X.shape[0])]
            for i in range(X.shape[0]):
                for j in range(self.m):
                    norms_test[i][j] = (np.linalg.norm(X[i] - self.X[j], 2)) ** 2
            gus = np.exp(-(np.array(norms_test)) * self.gamma)

        ans = np.array(gus) * np.squeeze(ya)
        ans = np.sum(ans, axis=1) + b
        ans[ans >= 0] = 1
        ans[ans != 1] = -1
        return ans

    # Update alphas
    def update_alphas(self):
        # Calculate the projection matrix
        a_mask = self.alphas.copy()
        a_mask[a_mask <= self.threshold] = 0
        # The mask of alphas is 1 if it is greater than the threshold, and 0 if it is less than the threshold
        a_mask[a_mask != 0] = 1

        pos = a_mask * self.y_pos
        count_pos = len(pos[pos != 0])
        sumgrads_pos = (self.d * pos).sum()

        neg = a_mask * self.y_neg
        count_neg = len(neg[neg != 0])
        sumgrads_neg = (self.d * neg).sum()

        res_pos = self.y_pos * (1 / count_pos) * (sumgrads_pos)
        res_neg = self.y_neg * (1 / count_neg) * (sumgrads_neg)
        res = res_pos + res_neg
        proj = self.d - res
        proj_d = a_mask * proj

        # The projected gradient is approximately equal to 0, and the threshold is 0.01
        while (np.linalg.norm(proj_d, np.inf) < 0.001):
            # Check whether the negative gradient component is included, if it is not included, the KKT condition is satisfied
            proj = proj.squeeze()
            w_index1 = np.argwhere(proj < 0)
            w_index2 = np.argwhere(a_mask.squeeze() == 0)
            w_index = np.intersect1d(w_index1.squeeze(), w_index2.squeeze())

            # Include negative components, recalculate the projection matrix
            if (len(w_index) != 0):
                a_mask[w_index] = 1

                # Recalculate the projection matrix and update the alphas corresponding to all negative components
                w_pos = w_index[np.argwhere(self.y_pos[w_index].squeeze() == 1)]

                if (len(w_pos) != 0):
                    count_pos += len(w_pos)
                    sumgrads_pos += np.sum(self.d[w_pos])
                    res_pos = self.y_pos * (1 / count_pos) * (sumgrads_pos)

                w_neg = w_index[np.argwhere(self.y_neg[w_index].squeeze() == 1)]
                if (len(w_neg) != 0):
                    count_neg += len(w_neg)
                    sumgrads_neg += np.sum(self.d[w_neg])
                    res_neg = self.y_neg * (1 / count_neg) * (sumgrads_neg)

                res = res_pos + res_neg
                proj_d = self.d - res
                proj_d = a_mask * proj_d

            else:
                # print("Proj KKT is statisfied.")
                self.stop = True
                break
        if (self.stop == False):
            # According to the upper and lower bounds of alpha, calculate the upper and lower bounds of the learning rate
            np.seterr(divide='ignore', invalid='ignore')
            lrs = self.alphas / proj_d
            lrs = min(lrs[lrs > 0])

            # Calculate the optimal learning rate
            extre_lrsup = np.dot(np.transpose(proj_d), proj_d).flatten()
            extre_lrsdown = np.dot(np.dot(np.transpose(proj_d), self.matrix), proj_d).flatten()
            extre_lrs = extre_lrsup / extre_lrsdown

            # Take a learning rate that does not violate the bounds
            new_lrs = min(extre_lrs, lrs)

            self.alphas = self.alphas - new_lrs * proj_d
            # Update alphas gradient
            d = np.dot(self.matrix, self.alphas).flatten()
            d = d[:, np.newaxis]
            self.d = d

        return self.d

    # Update gamma
    def update_gamma(self, gamma_change):
        m = np.array(self.matrix)
        n = np.array(self.norms)
        matrix = m * (-n)
        gamma_d = np.dot(self.alphas.transpose(1, 0), matrix)
        gamma_d = 0.5 * np.dot(gamma_d, self.alphas)

        # The learning rate of gamma
        l = gamma_change

        # According to the upper and lower bounds of gamma, calculate the upper and lower bounds of the learning rate
        if (gamma_d > 0):
            lr = l
        elif (gamma_d < 0):
            m = -(self.gamma) / (gamma_d[0][0] * 2)
            lr = min(l, m)
        else:
            lr = 0

        # Record the loss before update
        oldloss = self.loss()
        temps = self.gamma

        self.gamma = self.gamma + lr * gamma_d[0][0]

        # When the loss does not rise, the gamma learning rate is continuously halved to ensure that the loss rises
        newloss = self.loss()
        flag = 0
        while (newloss < oldloss and flag < 20):
            flag += 1
            lr = lr / 2
            self.gamma = temps + lr * gamma_d[0][0]
            newloss = self.loss()

        # After given iterations number, the loss still does not increase, stop the iteration
        if (flag >= self.lossIter):
            self.gamma = temps
            # print("gamma search can't decline loss")
            self.stop = True

        self.loss()
        return gamma_d[0][0]

    def loss(self):
        # Update the Gaussian kernel matrix
        gauss = np.exp(-(np.array(self.norms)) * self.gamma)
        self.gauss = gauss

        # Update the G matrix
        y = np.array(self.y).reshape(self.m, 1)
        ys = np.dot(y, np.transpose(y))
        matrix = gauss * ys
        self.matrix = matrix

        # Update alphas gradient
        d = np.dot(self.matrix + self.delt * np.eye(self.m), self.alphas).flatten()
        d = d[:, np.newaxis]
        self.d = d

        F = 0.5 * np.dot(self.alphas.transpose(1, 0), self.d)
        return F[0][0]

    def fit(self):
        # Initialize the norm matrix, ||xi-xj||^2
        norms = [[0] * self.m for _ in range(self.m)]
        for i in range(self.m):
            for j in range(self.m):
                norms[i][j] = (np.sum((self.X[i] - self.X[j]) ** 2))
        self.norms = norms

        # Initialize the Gaussian kernel matrix, e^(-gamma*||xi-xj||^2)
        gauss = np.exp(-(np.array(self.norms)) * self.gamma)
        self.gauss = gauss

        # Initialize the G matrix, yi*yj*e^(-gamma*||xi-xj||^2)
        y = np.array(self.y).reshape(self.m, 1)
        ys = np.dot(y, np.transpose(y))
        matrix = gauss * ys
        self.matrix = matrix + self.delt * np.eye(self.m)

        # Initialize the alphas gradient, (yi*yj*e^(-gamma*||xi-xj||^2)+delt)*alpha
        d = np.dot(self.matrix, self.alphas).flatten()
        d = d[:, np.newaxis]
        self.d = d

        # Initialize the learning rate of gamma to 1
        gamma_change = 1
        epochs = 0
        totalgammas = []

        while (epochs < self.gammaIter):
            count = 0

            # Update gamma
            self.update_gamma(gamma_change)
            totalgammas.append(self.gamma)
            self.stop = False

            while (count < self.max_epoch):
                # Update alphas
                self.update_alphas()

                count += 1
                if (self.stop == True):
                    break

            epochs += 1

            l = self.loss()
            # print("loss:", epochs, l)

            # Update gamma using dynamic learning rate
            if (epochs > 2):
                gamma_change = np.abs(totalgammas[-2] - totalgammas[-1])

            # Stop iteration
            if (epochs > 2 and np.abs(totalgammas[-2] - totalgammas[-1]) < self.stopaccuracy):
                break

    def predict(self, X):
        pre = self.decision_function(X)
        return pre
