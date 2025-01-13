#!pip install ripser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, jaccard_score, rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering, MeanShift, estimate_bandwidth, DBSCAN, OPTICS, AgglomerativeClustering

import networkx as nx
from sklearn.metrics import pairwise_distances
from itertools import combinations

#label = np.loadtxt('Glass-label.txt')
#data = np.loadtxt('glass1.txt')

#label = np.loadtxt('yeast-label.txt')
#data = np.loadtxt('yeast.txt')

label = np.loadtxt('usps-label.txt')
data = np.loadtxt('usps-1.txt')


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

"""#Compare with other clusterings

##k-means
"""

# Define K-means clustering
def kmeans_clustering(data, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_

# Run K-means clustering
kmeans_labels = kmeans_clustering(data, num_clusters=10)

"""##Spectral"""

# Define Spectral clustering
def spectral_clustering(data, num_clusters=3):
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
    spectral.fit(data)
    return spectral.labels_

# Run spectral clustering
spectral_labels = spectral_clustering(data, num_clusters=10)

"""##Tomato"""

#!pip install gudhi

# using ToMATo method
from gudhi.clustering.tomato import Tomato

def Tomato_clustering(data):
    t = Tomato()
    t.fit(data)
    t.n_clusters_= 10
    return t.labels_

# Run Tomato clustering
t_labels = Tomato_clustering(data)

"""##DBSCAN"""

# Define DBSCAN clustering
def dbscan_clustering(data, eps=0.5, min_samples=3):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data_scaled)
    return dbscan.labels_

#Run DBSCAN Clustering
dbscan_labels = dbscan_clustering(data, eps=0.5, min_samples=10)

"""##Agglomerative"""

## define Agglomerative Clustering
def agglomerative_clustering(data, num_clusters):
    agglomerative = AgglomerativeClustering(n_clusters=num_clusters, linkage='single')
    agglomerative.fit(data)
    return agglomerative.labels_

# Run Agglomerative Clustering
agglomerative_labels = agglomerative_clustering(data,num_clusters=10)

"""##Other clusterings"""

# Define MiniBatchkmeans clustering
def minibatchkmeans(data, num_clusters=2):
    minibatchkmeans= MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', max_iter=20, random_state=42)
    minibatchkmeans.fit(data)
    return minibatchkmeans.labels_

## define Optics Clustering
def optics(data):
    optics= OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.1)
    optics.fit(data)
    return optics.labels_

# Define Mean Shift Clustering
def mean_shift(data):
    mean_shift = MeanShift()
    mean_shift.fit(data)
    return mean_shift.labels_

"""##Run other clustering methods"""

# Run Minibatchkmeans
minibatchkmeans_labels = minibatchkmeans(data, num_clusters=10)

# Run Mean-Shift Clustering
mean_shift_labels = mean_shift(data)

# Run Optics
optics_labels = optics(data)

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, jaccard_score, rand_score

tomato_ari = adjusted_rand_score(label, t_labels)
tomato_ri = rand_score(label, t_labels)
tomato_nmi = normalized_mutual_info_score(label, t_labels)

spectral_ari = adjusted_rand_score(label, spectral_labels)
spectral_ri = rand_score(label, spectral_labels)
spectral_nmi = normalized_mutual_info_score(label, spectral_labels)

kmeans_ari = adjusted_rand_score(label, kmeans_labels)
kmeans_ri = rand_score(label, kmeans_labels)
kmeans_nmi = normalized_mutual_info_score(label, kmeans_labels)

dbscan_ari = adjusted_rand_score(label, dbscan_labels)
dbscan_ri = rand_score(label, dbscan_labels)
dbscan_nmi = normalized_mutual_info_score(label, dbscan_labels)

ac_ari = adjusted_rand_score(label, agglomerative_labels)
ac_ri = rand_score(label, agglomerative_labels)
ac_nmi = normalized_mutual_info_score(label, agglomerative_labels)

optics_ari = adjusted_rand_score(label, optics_labels)
optics_ri = rand_score(label, optics_labels)
optics_nmi = normalized_mutual_info_score(label, optics_labels)

minibatchkmeans_ari = adjusted_rand_score(label, minibatchkmeans_labels)
minibatchkmeans_ri = rand_score(label, minibatchkmeans_labels)
minibatchkmeans_nmi = normalized_mutual_info_score(label, minibatchkmeans_labels)

mean_shift_ari = adjusted_rand_score(label, mean_shift_labels)
mean_shift_ri = rand_score(label, mean_shift_labels)
mean_shift_nmi = normalized_mutual_info_score(label, mean_shift_labels)

print(f'BNFC RI: {bnfc_ri:.3f}, BNFC ARI: {bnfc_ari:.3f}, NMI: {bnfc_nmi: .3f}')
print(f'Agglomerative RI: {ac_ri:.3f}, Agglomerative ARI: {ac_ari:.3f}, NMI: {ac_nmi: .3f}')
print(f'Kmeans RI: {kmeans_ri:.3f}, Kmeans ARI: {kmeans_ari:.3f}, NMI: {kmeans_nmi: .3f}')
print(f'Spectral RI: {spectral_ri:.3f}, Spectral ARI: {spectral_ari:.3f}, NMI: {spectral_nmi: .3f}')
print(f'Tomato RI: {tomato_ri:.3f}, Tomato ARI: {tomato_ari:.3f},Tomato NMI: {tomato_nmi: .3f}')
print(f'DBSCAN RI: {dbscan_ri:.3f}, DBSCAN ARI: {dbscan_ari:.3f}, NMI: {dbscan_nmi: .3f}')
print(f'Mean-Shift RI: {mean_shift_ri:.3f}, Mean-Shift ARI: {mean_shift_ari:.3f}, NMI: {mean_shift_nmi: .3f}')
print(f'Minibatchkmeans RI: {minibatchkmeans_ri:.3f}, Minibatchkmeans ARI: {minibatchkmeans_ari:.3f}, NMI: {minibatchkmeans_nmi: .3f}')
print(f'Optics RI: {optics_ri:.3f}, Optics ARI: {optics_ari:.3f}, NMI: {optics_nmi: .3f}')


#Plotting the data
label = np.loadtxt('usps-label.txt')
data = np.loadtxt('usps_tsne.txt')

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = label
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'],s=60, cmap='viridis')
plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
plt.show()

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = bnfc_label
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'],s=60, cmap='viridis')
plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
plt.show()


data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = t_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'],s=60, cmap='viridis')
plt.show()

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = kmeans_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'],s=60, cmap='viridis')
plt.show()

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = spectral_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'],s=60, cmap='viridis')
plt.show()

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = dbscan_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'],s=60, cmap='viridis')
plt.show()

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = agglomerative_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'],s=60, cmap='viridis')
plt.show()

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = optics_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'],s=60, cmap='viridis')
plt.show()

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = minibatchkmeans_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'],s=60, cmap='viridis')
plt.show()

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = mean_shift_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'],s=60, cmap='viridis')
plt.show()

