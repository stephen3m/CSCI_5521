import numpy as np
# some tips
# |S| is the determinant of S in the discriminant functions, try np.linalg.det()
# you can also directly get the inverse of a matrix by np.linalg.inv()


# ------------------------------------- You are going to implement 3 classifiers and corresponding helper functions --------------------
# ------------------------------------- Three classifiers start from here --------------------------------------------------------------
class GaussianDiscriminant_C1:
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.S = np.zeros((k,d,d))   # S1 and S2, store in 2*(8*8) matrices
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1, S1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)
        # m2, S2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        # can be ignored, removed or replaced with any following implementations
        predictions = np.zeros(Xtest.shape[0])

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finall get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)
        g1,g2 = [],[]
        
        s1 = self.S[0,:,:]
        mean1 = self.m[0,:]
        big_w1 = -1*0.5*np.linalg.inv(s1)
        small_w1 = np.linalg.inv(s1).dot(mean1)
        w1_0 = ((-1*0.5*(mean1.T)).dot(np.linalg.inv(s1)).dot(mean1))-(0.5*np.log(np.linalg.det(s1)))+np.log(self.p[0])

        s2 = self.S[1,:,:]
        mean2 = self.m[1,:]
        big_w2 = -1*0.5*np.linalg.inv(s2)
        small_w2 = np.linalg.inv(s2).dot(mean2)
        w1_1 = (-1*0.5*(mean2.T).dot(np.linalg.inv(s2).dot(mean2)))-(0.5*np.log(np.linalg.det(s2)))+np.log(self.p[1])

        for x in Xtest:
            g1.append((x.T).dot(big_w1).dot(x) + (small_w1.T.dot(x)) + w1_0)
            g2.append((x.T).dot((big_w2).dot(x)) + (small_w2.T.dot(x)) + w1_1)

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step2: 
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        for i in range(len(predictions)):
            if(g1[i] > g2[i]):
                predictions[i] = 1
            else:
                predictions[i] = 2
        print(predictions)
        return predictions


class GaussianDiscriminant_C2:
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.S = np.zeros((k,d,d))  # S1 and S2, store in 2*(8*8) matrices
        self.shared_S =np.zeros((d,d))  # the shared convariance S that will be used for both classes
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1, S1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)
        # m2, S2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step 3: Compute the shared covariance matrix that is used for both class
        # shared_S = p1*S1+p2*S2
        self.shared_S = self.p[0]*self.S[0,:,:] + self.p[1]*self.S[1,:,:]


    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        # can be ignored, removed or replaced with any following implementations
        predictions = np.zeros(Xtest.shape[0])

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finall get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)
        g1, g2 = [], []
        
        m1 = self.m[0,:]
        m2 = self.m[1,:]
        shared_S = self.shared_S
        w1 = np.linalg.inv(shared_S).dot(m1)
        w1_0 = -0.5*m1.T.dot(np.linalg.inv(shared_S)).dot(m1) + np.log(self.p[0])
        w2 = np.linalg.inv(shared_S).dot(m2)
        w2_0 = -0.5*m2.T.dot(np.linalg.inv(shared_S)).dot(m2) + np.log(self.p[1])
        
        for x in Xtest:
            g1.append(w1.T.dot(x) + w1_0)
            g2.append(w2.T.dot(x) + w2_0)
                                                                      
        # s1 = self.S[0,:,:]
        # mean1 = self.m[0,:]
        # big_w1 = -1*0.5*np.linalg.inv(s1)
        # small_w1 = np.linalg.inv(s1).dot(mean1)
        # w1_0 = ((-1*0.5*(mean1.T)).dot(np.linalg.inv(s1)).dot(mean1))-(0.5*np.log(np.linalg.det(s1)))+np.log(self.p[0])

        # s2 = self.S[1,:,:]
        # mean2 = self.m[1,:]
        # big_w2 = -1*0.5*np.linalg.inv(s2)
        # small_w2 = np.linalg.inv(s2).dot(mean2)
        # w1_1 = (-1*0.5*(mean2.T).dot(np.linalg.inv(s2).dot(mean2)))-(0.5*np.log(np.linalg.det(s2)))+np.log(self.p[1])

        # for x in Xtest:
        #     g1.append((x.T).dot(big_w1).dot(x) + (small_w1.T.dot(x)) + w1_0)
        #     g2.append((x.T).dot((big_w2).dot(x)) + (small_w2.T.dot(x)) + w1_1)

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step2: 
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]
        for i in range(len(predictions)):
            if(g1[i] > g2[i]):
                predictions[i] = 1
            else:
                predictions[i] = 2
        print(predictions)
        return predictions


class GaussianDiscriminant_C3:
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.S = np.zeros((k,d,d))  # S1 and S2, store in 2*(8*8) matrices
        self.shared_S =np.zeros((d,d))  # the shared convariance S that will be used for both classes
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1, S1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)
        # m2, S2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step 3: Compute the diagonal of S1 and S2, since we assume each feature is independent, and has non diagonal entries cast to 0
        # [[1,2],[2,4]] => [[1,0],[0,4]], try np.diag() twice
        self.S[0,:,:] = np.diag(np.diag(self.S[0,:,:]))
        self.S[1,:,:] = np.diag(np.diag(self.S[1,:,:]))

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step 4: Compute the shared covariance matrix that is used for both class
        # shared_S = p1*S1+p2*S2
        self.shared_S = self.p[0]*self.S[0,:,:] + self.p[1]*self.S[1,:,:]

    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        # can be ignored, removed or replaced with any following implementations
        predictions = np.zeros(Xtest.shape[0])

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finall get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)
        # Please note here, currently we assume shared_S is a d*d diagonal matrix, the non-capital si^2 in the lecture formula will be the i-th entry on the diagonal
        g1, g2 = [], []

        variance_vals = np.diag(self.shared_S)

        for x in Xtest:
            g1.append(-0.5 * np.sum((x - self.m[0,:])**2 / variance_vals) + np.log(self.p[0]))
            g2.append(-0.5 * np.sum((x - self.m[1,:])**2 / variance_vals) + np.log(self.p[1]))

        # s1 = self.S[0,:,:]
        # mean1 = self.m[0,:]
        # big_w1 = -1*0.5*np.linalg.inv(s1)
        # small_w1 = np.linalg.inv(s1).dot(mean1)
        # w1_0 = ((-1*0.5*(mean1.T)).dot(np.linalg.inv(s1)).dot(mean1))-(0.5*np.log(np.linalg.det(s1)))+np.log(self.p[0])

        # s2 = self.S[1,:,:]
        # mean2 = self.m[1,:]
        # big_w2 = -1*0.5*np.linalg.inv(s2)
        # small_w2 = np.linalg.inv(s2).dot(mean2)
        # w1_1 = (-1*0.5*(mean2.T).dot(np.linalg.inv(s2).dot(mean2)))-(0.5*np.log(np.linalg.det(s2)))+np.log(self.p[1])

        # for x in Xtest:
        #     g1.append((x.T).dot(big_w1).dot(x) + (small_w1.T.dot(x)) + w1_0)
        #     g2.append((x.T).dot((big_w2).dot(x)) + (small_w2.T.dot(x)) + w1_1)

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step2: 
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]
        for i in range(len(predictions)):
            if(g1[i] > g2[i]):
                predictions[i] = 1
            else:
                predictions[i] = 2
        print(predictions)
        return predictions


# ------------------------------------- Helper Functions start from here --------------------------------------------------------------
# Input:
# features: n*d matrix (n is the number of samples, d is the number of dimensions of the feature)
# labels: n vector
# Output:
# features1: n1*d
# features2: n2*d
# n1+n2 = n, n1 is the number of class1, n2 is the number of samples from class 2
def splitData(features, labels):
    # placeholders to store the separated features (feature1, feature2), 
    # can be ignored, removed or replaced with any following implementations
    features1 = features[labels==1]  
    features2 = features[labels==2]

    # fill in the code here !!!!!!!!!!!!!!!!!!!!!!!
    # separate the features according to the corresponding labels, for example
    # if features = [[1,1],[2,2],[3,3],[4,4]] and labels = [1,1,1,2], the resulting feature1 and feature2 will be
    # feature1 = [[1,1],[2,2],[3,3]], feature2 = [[4,4]]
    return features1, features2


# compute the mean of input features
# input: 
# features: n*d
# output: d
def computeMean(features):
    # placeholders to store the mean for one class
    # can be ignored, removed or replaced with any following implementations
    # m = np.zeros(features.shape[1])

    # fill in the code here !!!!!!!!!!!!!!!!!!!!!!! 
    # try to explore np.mean() for convenience
    return np.mean(features, axis=0)


# compute the mean of input features
# input: 
# features: n*d
# output: d*d
def computeCov(features):
    # placeholders to store the covariance matrix for one class
    # can be ignored, removed or replaced with any following implementations
    # S = np.eye(features.shape[1])

    # fill in the code here !!!!!!!!!!!!!!!!!!!!!!!
    # try to explore np.cov() for convenience
    return np.cov(features.T)


# compute the priors of input features
# input: 
# features: n
# output: 2
def computePrior(labels):
    # placeholders to store the priors for both class
    # can be ignored, removed or replaced with any following implementations
    p = np.array([0.5,0.5])

    # fill in the code here !!!!!!!!!!!!!!!!!!!!!!! 
    # p1 = numOf class1 / numOf all the data; same as p2
    p1_count = 0
    p2_count = 0
    for i in range(len(labels)):
        if(labels[i] == 1):
            p1_count += 1
        elif(labels[i] == 2):
            p2_count += 1
    p[0] = p1_count/len(labels)
    p[1] = p2_count/len(labels)
    return p
