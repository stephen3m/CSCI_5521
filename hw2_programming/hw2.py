# import libraries
import numpy as np
from matplotlib import pyplot as plt
from Mykmeans import Kmeans
from MyPCA import PCA
import time

# read in data.
data=np.genfromtxt("Digits089.csv",delimiter=",")
Xtrain=data[data[:,0]!=5,2:]
ytrain=data[data[:,0]!=5,1]
Xtest=data[data[:,0]==5,2:]
ytest=data[data[:,0]==5,1]

# ---------------------------------------- Question (a) -------------------------------------------------
# --------- Considering K = 8 ---------
# apply kmeans algorithms to raw data
clf = Kmeans(k=8)
start = time.time()
num_iter, error_history = clf.fit(Xtrain, ytrain)
time_raw = time.time() - start

# plot the history of reconstruction error
fig = plt.figure()
plt.plot(np.arange(len(error_history)),error_history,'b-',linewidth=2)
fig.set_size_inches(10, 10)
fig.savefig('raw_data_K8.png')
plt.show()

# using kmeans clustering for classification
predicted_label = clf.predict(Xtest)
acc_raw_K8 = np.count_nonzero(predicted_label==ytest)/len(Xtest)

# --------- Considering K = 4 ---------
# apply kmeans algorithms to raw data
clf_new = Kmeans(k=4)
start_new = time.time()
num_iter_new, error_history_new = clf_new.fit(Xtrain, ytrain)
time_raw_new = time.time() - start

# plot the history of reconstruction error
fig = plt.figure()
plt.plot(np.arange(len(error_history_new)),error_history_new,'b-',linewidth=2)
fig.set_size_inches(10, 10)
fig.savefig('raw_data_K4.png')
plt.show()

# using kmeans clustering for classification
predicted_label = clf_new.predict(Xtest)
acc_raw_K4 = np.count_nonzero(predicted_label==ytest)/len(Xtest)

# ---------------------------------------- Question (b) -------------------------------------------------
# apply kmeans algorithms to low-dimensional data (PCA) that captures >95% of variance
pca = PCA(0.95)
Xtrain_pca, num_dim = pca.fit(Xtrain)
clf = Kmeans(k=8)
start = time.time()
num_iter_pca, error_history_pca = clf.fit(Xtrain_pca, ytrain)
time_pca = time.time() - start

# plot the history of reconstruction error
fig1 = plt.figure()
plt.plot(np.arange(len(error_history_pca)),error_history_pca,'b-',linewidth=2)
fig1.set_size_inches(10, 10)
fig1.savefig('pca.png')
plt.show()

# using kmeans clustering for classification
Xtest_pca = pca.predict(Xtest)
predicted_label = clf.predict(Xtest_pca)
acc_pca = np.count_nonzero(predicted_label==ytest)/len(Xtest)

# ---------------------------------------- Question (b) -------------------------------------------------
# apply kmeans algorithms to 1D feature obtained from PCA
pca = PCA(0, num_dim=1)
Xtrain_pca, _ = pca.fit(Xtrain)
clf = Kmeans(k=8)
start = time.time()
num_iter_pca_1, error_history_pca_1 = clf.fit(Xtrain_pca, ytrain)
time_pca_1 = time.time() - start

# plot the history of reconstruction error
fig2 = plt.figure()
plt.plot(np.arange(len(error_history_pca_1)),error_history_pca_1,'b-',linewidth=2)
fig2.set_size_inches(10, 10)
fig2.savefig('pca_1d.png')
plt.show()

# using kmeans clustering for classification
Xtest_pca = pca.predict(Xtest)
predicted_label = clf.predict(Xtest_pca)
acc_pca_1 = np.count_nonzero(predicted_label==ytest)/len(Xtest)
# Printing reconstruction error for K = 8
print('Reconstruction error for K=8:', error_history[-1])

# Printing reconstruction error for K = 4
print('Reconstruction error for K=4:', error_history_new[-1])

# # print the average information entropy and number of iterations for convergence
print('Using raw data with K=8 converged in %d iteration (%.2f seconds)' % (num_iter,time_raw))
print('Classification accuracy: %.2f' %acc_raw_K8)

print('#################')
print('Using raw data with K=4 converged in %d iteration (%.2f seconds)' % (num_iter_new,time_raw_new))
print('Classification accuracy: %.2f' %acc_raw_K4)

print('#################')
print('Project data into %d dimensions with PCA converged in %d iteration (%.2f seconds)' % (num_dim,num_iter_pca,time_pca))
print('Classification accuracy: %.2f' %acc_pca)

print('#################')
print('Project data into 1 dimension with PCA converged in %d iteration (%.2f seconds)' % (num_iter_pca_1,time_pca_1))
print('Classification accuracy: %.2f' %acc_pca_1)
