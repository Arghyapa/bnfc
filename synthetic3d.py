# Package Import
from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering, MeanShift, estimate_bandwidth, DBSCAN, OPTICS, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse.csgraph import connected_components

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, jaccard_score, rand_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
import networkx as nx
from sklearn.metrics import pairwise_distances
from itertools import combinations

"""#generate Torus dataset"""

pi = np.pi
torusnum1=1000
torusnum2=1000
linenum1=200
linenum2=200
linenum3=200
u_1 = np.random.uniform(0, 2*pi, torusnum1)
v_1 = np.random.uniform(0, 2*pi, torusnum1)
x_1 = (5 + 2 * np.cos(v_1)) * np.cos(u_1)
y_1 = (5 + 2 * np.cos(v_1)) * np.sin(u_1)
z_1 = 2 * np.sin(v_1)
D_1 = np.zeros((torusnum1, 4))
for i in range(0, torusnum1):
    D_1[i][0] = x_1[i]
    D_1[i][1] = y_1[i]
    D_1[i][2] = z_1[i]
    D_1[i][3] = 0

u_2 = np.random.uniform(0, 2*pi, torusnum2)
v_2 = np.random.uniform(0, 2*pi, torusnum2)
x_2 = (5 + 2 * np.cos(v_2)) * np.cos(u_2) + 20
y_2 = (5 + 2 * np.cos(v_2)) * np.sin(u_2)
z_2 = 2 * np.sin(v_2)
D_2 = np.zeros((torusnum2, 4))
for i in range(0, torusnum2):
    D_2[i][0] = x_2[i]
    D_2[i][1] = y_2[i]
    D_2[i][2] = z_2[i]
    D_2[i][3] = 1

x_3 = np.random.uniform(0, 20, linenum1)
y_3 = np.zeros((linenum1,1)) + 7
z_3 = np.zeros((linenum1,1))
D_3 = np.zeros((linenum1, 4))
for i in range(0, linenum1):
    D_3[i][0] = x_3[i]
    D_3[i][1] = y_3[i]
    D_3[i][2] = z_3[i]
    D_3[i][3] = 2

x_4 = np.random.uniform(0, 20, linenum2)
y_4 = np.zeros((linenum2,1)) - 7
z_4 = np.zeros((linenum2,1))
D_4 = np.zeros((linenum2, 4))
for i in range(0, linenum2):
    D_4[i][0] = x_4[i]
    D_4[i][1] = y_4[i]
    D_4[i][2] = z_4[i]
    D_4[i][3] = 2
x_5 = np.zeros((linenum3,1)) + 10
y_5 = np.random.uniform(-7, 7, linenum3)
z_5 = np.zeros((linenum3,1))
D_5 = np.zeros((linenum3, 4))
for i in range(0, linenum3):
    D_5[i][0] = x_5[i]
    D_5[i][1] = y_5[i]
    D_5[i][2] = z_5[i]
    D_5[i][3] = 3

D = np.concatenate((D_1, D_2, D_3, D_4, D_5), axis=0)
label = D[:,3]
data = D[:,0:3]

np.random.seed(42)
x = data[:,0]
y = data[:,1]
z = data[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_axis_off()
#ax.set_title('2 torus with 3 Lines')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=label, cmap='viridis', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
#ax.set_title('Ground Truth')
ax.set_axis_off()
plt.show()

"""#BNFC Clustering"""

#!pip install gudhi

#!pip install ripser

from ripser import ripser

num_clust = int(input("Number of Clusters:"))

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def extract_betti(PD, scale_seq):
    b = np.zeros(len(scale_seq))
    for k in range(len(scale_seq)):
        b[k] = np.sum((scale_seq[k] >= PD[:, 0]) & (scale_seq[k] < PD[:, 1]))
    return b

def weighted_knn(data, nKNN):
    knn = NearestNeighbors(n_neighbors=nKNN).fit(data)
    dists, ind = knn.kneighbors(data)

    # Compute weights inversely proportional to distances
    weights = 1 / (dists + 1e-5)  # Adding a small value to avoid division by zero

    return ind, weights

def bnfc_weighted(data, nKNN, filt_len=100):
    N = data.shape[0]

    # Perform weighted k-NN
    ind, weights = weighted_knn(data, nKNN)  # Get indices and weights from weighted knn
    maxscale = np.max(1 / weights[:, -1])  # Max of inverse of weights (distances)

    scale_seq = np.linspace(0, maxscale, filt_len)

    # Initialize arrays to store Betti numbers
    betti_0 = np.zeros((N, filt_len))
    betti_1 = np.zeros((N, filt_len))
    betti_2 = np.zeros((N, filt_len))

    for i in range(N):

        neighborhood = data[ind[i,]]

        # Compute persistent homology
        rips = ripser(neighborhood, maxdim=2)
        PD_0 = rips['dgms'][0] 
        PD_1 = rips['dgms'][1]
        PD_2 = rips['dgms'][2]

        # Replace infinite persistence values with maxscale for Betti-0 & Betti-1
        PD_0[PD_0[:, 1] == np.inf, 1] = maxscale
        PD_1[PD_1[:, 1] == np.inf, 1] = maxscale
        PD_2[PD_2[:, 1] == np.inf, 1] = maxscale

        # Extract Betti-0 and Betti-1 numbers for this point over the scale sequence
        betti_0[i, :] = extract_betti(PD_0, scale_seq)
        betti_1[i, :] = extract_betti(PD_1, scale_seq)
        betti_2[i, :] = extract_betti(PD_2, scale_seq)

    cos_scores_0= np.zeros((N, nKNN))
    cos_scores_1= np.zeros((N, nKNN))
    cos_scores_2= np.zeros((N, nKNN))

    # Compute cosine similarity between data points based on their Betti numbers
    for i in range(N):
        # Get Betti numbers for the current point and its neighbors
        current_betti_0 = betti_0[i, :].reshape(1, -1)
        current_betti_1 = betti_1[i, :].reshape(1, -1)
        current_betti_2 = betti_2[i, :].reshape(1, -1)

        neighbors_betti_0 = betti_0[ind[i,], :]
        neighbors_betti_1 = betti_1[ind[i,], :]
        neighbors_betti_2 = betti_2[ind[i,], :]

        # Compute cosine similarity for Betti-0, Betti-1, and Betti-2 arrays
        sim_0 = cosine_similarity(current_betti_0, neighbors_betti_0).flatten()
        cos_scores_0[i,:]= sim_0

        sim_1 = cosine_similarity(current_betti_1, neighbors_betti_1).flatten()
        cos_scores_1[i,:]= sim_1

        sim_2 = cosine_similarity(current_betti_2, neighbors_betti_2).flatten()
        cos_scores_2[i,:]= sim_2

    # Initialize adjacency matrix
    A = np.zeros((N, N))

    # Calculate thresholds using the IQR method (Interquartile Range)
    tao_0 = np.percentile(cos_scores_0, 25) - 1.5 * (np.percentile(cos_scores_0, 75) - np.percentile(cos_scores_0, 25))
    meu_0 = np.percentile(cos_scores_1, 25) - 1.5 * (np.percentile(cos_scores_1, 75) - np.percentile(cos_scores_1, 25))
    gamma_0 = np.percentile(cos_scores_2, 25) - 1.5 * (np.percentile(cos_scores_2, 75) - np.percentile(cos_scores_2, 25))

    for i in range(N):
        indices_1 = ind[i, cos_scores_0[i,] >= tao_0]
        indices_2 = ind[i, cos_scores_1[i,] >= meu_0]
        indices_3 = ind[i, cos_scores_2[i,] >= gamma_0]


        # Find the common neighbors that meet the threshold in all three Betti dimensions
        common_indices = np.intersect1d(np.intersect1d(indices_1, indices_2), indices_3)

        # Form adjacency matrix by connecting similar points
        A[i, common_indices] = 1

    return A

from numpy.linalg import eig
rng = np.random.default_rng()

def tpspect(data, nKNN):

  def calculate_U(X, V):
      n, d = X.shape
      U = np.zeros(n)
      for i in range(n):
          dist = ((X[i] - V) * (X[i] - V)).sum(1)
          j = dist.argmin()
          U[i] =j
      return U

  def calculate_V(X, U, k):
      n, d = X.shape
      V = np.zeros([k, d])
      for j in range(k):
          index = np.where(U == j)[0]
          r = index.shape[0]
          if r == 0:
              V[j] = rng.choice(X)
          else:
              for i in index:
                  V[j] = V[j] + X[i]
              V[j] = V[j] / r
      return V

  def k_means(X, k):
      n, d = X.shape
      U = np.zeros(n)
      V = rng.choice(X, k)
      condition = True
      while (condition):
          temp = np.copy(U)
          U = calculate_U(X, V)
          V = calculate_V(X, U, k)
          condition = not np.array_equal(U, temp)
      return U

  k = num_clust
  n = data.shape[0]

  A = bnfc_weighted(data, nKNN)

  Adj = np.zeros((n, n))
  sigma = np.std(data)
  for i in range(n):
    for j in range(n):
      if A[i, j]==1:
        Adj[i, j] = np.exp(-np.linalg.norm(data[i] - data[j])**2 / (2*(sigma**2)))*A[i,j]*A[j,i]

  D = np.diag(np.sum(Adj, axis=1))

  L = D - Adj
  eigenvalues, eigenvectors = eig(L)

  indices = np.argsort(eigenvalues)[:k]
  U = eigenvectors[:,indices]

  bnfc_label = k_means(U,k)

  return bnfc_label

"""#Plot ARI vs nKNN"""

# List to store ARI scores
ari_scores = []

# Range of nKNN values to test
nKNN_values = range(1, 15)  # Adjust the range as needed

# Perform clustering and compute ARI for each nKNN
for nKNN in nKNN_values:
    Tplabel = tpspect(data, nKNN)
    ari = adjusted_rand_score(label, Tplabel)
    ari_scores.append(ari)

# Plot the ARI scores
plt.figure(figsize=(10, 6))
plt.plot(nKNN_values, ari_scores, marker='o')
plt.xlabel('nKNN')
plt.ylabel('Adjusted Rand Index (ARI)')
plt.title('ARI for Different nKNN Values')
plt.grid(True)
plt.show()

KNN = int(input("KNN:"))
#KNN = np.argmax(ari_scores)+1
bnfc_label = tpspect(data, nKNN=KNN)

bnfc_ari = adjusted_rand_score(label, bnfc_label)
bnfc_ri = rand_score(label, bnfc_label)
bnfc_nmi = normalized_mutual_info_score(label, bnfc_label)

print(f'BNFC RI: {bnfc_ri:.3f}, BNFC ARI: {bnfc_ari:.3f}, NMI: {bnfc_nmi: .3f}')

# Plotting the noisy data
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z,c= bnfc_label, marker='o', cmap='viridis')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
#ax.set_title('2 Torus 3 Line with Noise')
plt.show()

'''
# # List to store RI scores
# ri_scores = []

# # Range of nKNN values to test
# nKNN_values = range(4, 12)  # Adjust the range as needed

# # Perform clustering and compute RI for each nKNN
# for nKNN in nKNN_values:
#     cbn_labels = CBN(dMatrix, nKNN=nKNN, dist_matrix=True)
#     ri = rand_score(label, cbn_labels)
#     ri_scores.append(ri)

# # Plot the RI scores
# plt.figure(figsize=(10, 6))
# plt.plot(nKNN_values, ri_scores, marker='o')
# plt.xlabel('nKNN')
# plt.ylabel('Rand Index (RI)')
# plt.title('RI for Different nKNN Values')
# plt.grid(True)
# plt.show()



"""#Data with Noise"""

import numpy as np
import matplotlib.pyplot as plt

# Add noise
np.random.seed(42)
noise_strength = 0.10  # Add gaussian noise
noise = np.random.normal(0, noise_strength, data.shape)
data_noisy = data + noise

# Plotting the noisy data
x = data_noisy[:, 0]
y = data_noisy[:, 1]
z = data_noisy[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('2 Torus 3 Line with Noise')
plt.show()

# Plotting the noisy data
x = data_noisy[:, 0]
y = data_noisy[:, 1]
z = data_noisy[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z,c= label, marker='o', cmap='viridis')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('2 Torus 3 Line with Noise')
plt.show()

# List to store ARI scores
ari_scores = []

# Range of nKNN values to test
nKNN_values = range(1, 15)

# Perform clustering and compute ARI for each nKNN
for nKNN in nKNN_values:
    bnfc_noise_labels = bnfc_weighted(data_noisy, nKNN=nKNN)
    ari = adjusted_rand_score(label, bnfc_noise_labels)
    ari_scores.append(ari)

# Plot the ARI scores
plt.figure(figsize=(10, 6))
plt.plot(nKNN_values, ari_scores, marker='o')
plt.xlabel('nKNN')
plt.ylabel('Adjusted Rand Index (ARI)')
plt.title('ARI for Different nKNN Values for Noise')
plt.grid(True)
plt.show()

bnfc_noise_labels = bnfc_weighted(data_noisy, nKNN=8)

np.random.seed(42)

# Plotting the noisy data
x = data_noisy[:, 0]
y = data_noisy[:, 1]
z = data_noisy[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=bnfc_noise_labels, cmap='viridis', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('2 Torus 3 Line with Noise')
plt.show()

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, jaccard_score, rand_score

bnfc_ari = adjusted_rand_score(label, bnfc_noise_labels)
bnfc_ri = rand_score(label, bnfc_noise_labels)
bnfc_nmi = normalized_mutual_info_score(label, bnfc_noise_labels)

print(f'BNFC Noise RI: {bnfc_ri:.3f}, BNFC Noise ARI: {bnfc_ari:.3f}, NMI: {bnfc_nmi: .3f}')

"""#Plot 2 ARI values in one plot"""

# List to store ARI scores for dnoise and dMatrix
ari_scores_dnoise = []
ari_scores_dMatrix = []

# Range of nKNN values to test
nKNN_values = range(1, 12)  # Adjust the range as needed

# Perform clustering and compute ARI for each nKNN for dnoise
for nKNN in nKNN_values:
    cbn_noise_labels = CBN(dnoise, nKNN=nKNN, dist_matrix=True)
    ari_dnoise = adjusted_rand_score(label, cbn_noise_labels)
    ari_scores_dnoise.append(ari_dnoise)

# Perform clustering and compute ARI for each nKNN for dMatrix
for nKNN in nKNN_values:
    cbn_labels = CBN(dMatrix, nKNN=nKNN, dist_matrix=True)
    ari_dMatrix = adjusted_rand_score(label, cbn_labels)
    ari_scores_dMatrix.append(ari_dMatrix)

# Plot the ARI scores
plt.figure(figsize=(10, 6))
plt.plot(nKNN_values, ari_scores_dnoise, marker='o', label='noise', color='blue')
plt.plot(nKNN_values, ari_scores_dMatrix, marker='s', label='data', color='red')
plt.xlabel('nKNN')
plt.ylabel('Adjusted Rand Index (ARI)')
plt.title('ARI for Different nKNN Values')
plt.legend()
plt.grid(True)
plt.show()

"""#ToMATo Clustering"""

# using ToMATo method
from gudhi.clustering.tomato import Tomato

# define ToMATo clustering
def Tomato_clustering(data):
    t = Tomato()
    t.fit(data)
    t.n_clusters_= 4
    return t.labels_

# Run Tomato clustering
t_labels = Tomato_clustering(data)

np.random.seed(42)
x = data[:,0]
y = data[:,1]
z = data[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=t_labels, cmap='viridis', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('ToMATo')

plt.show()

"""#K-Means Clustering"""

# Define K-means clustering
def kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_

# Run K-means clustering
kmeans_labels = kmeans_clustering(data, num_clusters=4)

np.random.seed(42)
x = data[:,0]
y = data[:,1]
z = data[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=kmeans_labels, cmap='viridis', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Kmeans')

plt.show()

"""#Spectral clustering"""

# Define Spectral clustering
def spectral_clustering(data, num_clusters):
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
    spectral.fit(data)
    return spectral.labels_

# Run spectral clustering
spectral_labels = spectral_clustering(data, num_clusters=4)

np.random.seed(42)
x = data[:,0]
y = data[:,1]
z = data[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=spectral_labels, cmap='viridis', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Spectral')
plt.show()

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, jaccard_score, rand_score

tomato_ari = adjusted_rand_score(label, t_labels)
tomato_ri = rand_score(label, t_labels)
tomato_nmi = normalized_mutual_info_score(label, t_labels)

spectral_ri = adjusted_rand_score(label, spectral_labels)
spectral_ari = rand_score(label, spectral_labels)
spectral_nmi = normalized_mutual_info_score(label, spectral_labels)

kmeans_ri = adjusted_rand_score(label, kmeans_labels)
kmeans_ari = rand_score(label, kmeans_labels)
kmeans_nmi = normalized_mutual_info_score(label, kmeans_labels)

print(f'BNFC RI: {bnfc_ri:.3f}, BNFC ARI: {bnfc_ari:.3f}, NMI: {bnfc_nmi: .3f}')
print(f'Kmeans RI: {kmeans_ri:.3f}, Kmeans ARI: {kmeans_ari:.3f}, NMI: {kmeans_nmi: .3f}')
print(f'Spectral RI: {spectral_ri:.3f}, Spectral ARI: {spectral_ari:.3f}, NMI: {spectral_nmi: .3f}')
print(f'Tomato RI: {tomato_ri:.3f}, Tomato ARI: {tomato_ari:.3f},Tomato NMI: {tomato_nmi: .3f}')

"""# TPCC"""

import numpy as np
import gudhi as gd
from pylab import *
from matplotlib import pyplot as plt
import scipy
import scipy.sparse
import scipy.io
import sklearn
import pandas as pd
import plotly.express as px
import sklearn.cluster
import sklearn.metrics
import seaborn as sns

sns.set_theme()

repeats=1 #100 in real experiments
res=15
num_steps= 1 # 16
# true_values=[122, 19, 126]

"""## dataset"""

true_labels = [100,60,100,60]
def make_points(noise=0):
    pi = np.pi
    spherenum=100
    circlenum=30
    circlenum1=37
    r_1 = np.random.uniform(0, 1, spherenum)
    theta_1 = np.random.uniform(-pi, pi, spherenum)
    phi_1 = np.random.uniform(0, pi, spherenum)
    x_1 = r_1*np.cos(phi_1)
    y_1 = r_1*np.sin(phi_1)*np.cos(theta_1)
    z_1 = r_1*np.sin(phi_1)*np.sin(theta_1)
    D_1 = np.zeros((spherenum, 4))
    for i in range(0, spherenum):
        D_1[i][0] = x_1[i]
        D_1[i][1] = y_1[i]
        D_1[i][2] = z_1[i]
        D_1[i][3] = 0

    theta_2 = np.random.uniform(-pi, pi, circlenum)
    x_2 = np.cos(theta_2) + 2
    y_2 = np.sin(theta_2)
    z_2 = np.zeros((circlenum,1))
    D_2 = np.zeros((circlenum, 4))
    for i in range(0, circlenum):
        D_2[i][0] = x_2[i]
        D_2[i][1] = y_2[i]
        D_2[i][2] = z_2[i]
        D_2[i][3] = 1

    r_3 = np.random.uniform(0, 1, spherenum)
    theta_3 = np.random.uniform(-pi, pi, spherenum)
    phi_3 = np.random.uniform(0, pi, spherenum)
    x_3 = r_3*np.cos(phi_3) + 4
    y_3 = r_3*np.sin(phi_3)*np.cos(theta_3)
    z_3 = r_3*np.sin(phi_3)*np.sin(theta_3)
    D_3 = np.zeros((spherenum, 4))
    for i in range(0, spherenum):
        D_3[i][0] = x_3[i]
        D_3[i][1] = y_3[i]
        D_3[i][2] = z_3[i]
        D_3[i][3] = 2

    theta_4 = np.random.uniform(-pi, pi, circlenum1)
    x_4 = np.cos(theta_4) + 6
    y_4 = np.sin(theta_4)
    z_4 = np.zeros((circlenum1,1))
    D_4 = np.zeros((circlenum1, 4))
    for i in range(0, circlenum1):
        D_4[i][0] = x_4[i]
        D_4[i][1] = y_4[i]
        D_4[i][2] = z_4[i]
        D_4[i][3] = 3

    D = np.concatenate((D_1, D_2, D_3, D_4), axis=0)
    label = D[:,3]
    data = D[:,0:3]


    np.random.seed(42)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    # print(type(data))
    # print(label)
    # return (len(D_1),len(D_2),len(D_3),len(D_4))
    return data

"""## aux functions"""

def get_simplices(simplicial_tree):
    maxdim = simplicial_tree.dimension()
    simplices = []
    for i in range(maxdim+1):
        simplices.append([])
    for simplextuple in simplicial_tree.get_simplices():
        simplex = simplextuple[0]
        simplices[len(simplex)-1].append(simplex)
    return simplices

def num_k_simplices(simplicial_tree, k):
    if k > 0:
        n = len(list(simplicial_tree.get_skeleton(k))) - \
            len(list(simplicial_tree.get_skeleton(k-1)))
    else:
        n = simplicial_tree.num_vertices()
    return n

def build_simplex_dict(simplicial_tree, simplices):
    maxdim = simplicial_tree.dimension()
    num_k_simplices_in_p = []
    simplexdict = []
    for i in range(maxdim+1):
        num = num_k_simplices(simplicial_tree, i)
        num_k_simplices_in_p.append(num)
        #print('Number of '+str(i)+'-simplices: '+str(num))
        simplexdict.append(dict(
            zip([str(simplex) for simplex in simplices[i]], range(num_k_simplices_in_p[i]))))
    return num_k_simplices_in_p, simplexdict


def extract_boundary_operators(simplices, simplexdict, num_k_simplices_in_p):
    maxdim = len(num_k_simplices_in_p)-1
    boundary_operators = []
    for k in range(maxdim):
        newmatrix = scipy.sparse.coo_matrix(
            (num_k_simplices_in_p[k], num_k_simplices_in_p[k+1]))
        coordi = []
        coordj = []
        entries = []
        for simplex in simplices[k+1]:
            simplex_index = simplexdict[k+1][str(simplex)]
            for i in range(k+2):
                new_simplex = simplex.copy()
                new_simplex.pop(i)
                new_simplex_index = simplexdict[k][str(new_simplex)]
                coordi.append(new_simplex_index)
                coordj.append(simplex_index)
                if i % 2 == 0:
                    entries.append(1)
                else:
                    entries.append(-1)
        boundary_operators.append(scipy.sparse.csc_matrix((np.array(entries), (np.array(coordi), np.array(
            coordj))), shape=(num_k_simplices_in_p[k], num_k_simplices_in_p[k+1]), dtype=float))
    return boundary_operators

def Hodge_Laplacian(boundary_operators, k):
    if k == len(boundary_operators):
        Bkm = boundary_operators[k-1]
        A = Bkm.transpose()@Bkm
    elif k > 0:
        Bk = boundary_operators[k]
        Bkm = boundary_operators[k-1]
        A = Bk@Bk.transpose()+Bkm.transpose()@Bkm
    else:
        Bk = boundary_operators[k]
        A = Bk@Bk.transpose()
    return A

def cluster_zeros(vecs, threshold, mean_norm):
    clusters = []
    for i in range(len(vecs)):
        if np.linalg.norm(vecs[i]) > threshold*mean_norm:
            clusters.append(1)
        else:
            clusters.append(0)
    return clusters

def normalise_point_wise_clustered_points(clustered_points_orig, incident_k_simplices, cluster_dim_indicator, total_num_clusters):
    clustered_points_new = np.array(clustered_points_orig)
    n, m = np.array(incident_k_simplices.shape)
    for index in range(len(clustered_points_new)):
        point = clustered_points_new[index]
        for i in range(total_num_clusters):
            clustered_points_new[index, i] = point[i] / \
                incident_k_simplices[index, int(
                    cluster_dim_indicator[i])]
    return clustered_points_new

def normalise_cluster_wise_clustered_points(clustered_points_orig):
    dataframe = pd.DataFrame(data=clustered_points_orig)
    means = dataframe.mean(axis=0)
    standard_deviations = dataframe.std(axis=0)
    means = np.array(means).T
    sds = np.array(standard_deviations).T
    for i in range(size(sds)):
        if sds[i] == 0:
            sds[i] = 1
    clustered_points_new = clustered_points_orig-means
    clustered_points_new = clustered_points_new/sds
    return clustered_points_new

"""## TPCC algorithm"""

'''
