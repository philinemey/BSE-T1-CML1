#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 01:03:44 2021

@author: Philine
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd



### Determining the optimal number of clusters ###

# Elbow method using inertias
def elbow(X_scaled, n):
    from sklearn.cluster import KMeans
    from kneed import KneeLocator
    
    SEED=10
    
    # instantiate and fit Kmeans to data
    # make 'Elbow graph' to deduce optimal number of clusters
    inertias = [] 
    K = range(2, n)
      
    for k in K: 
        #Building and fitting the model 
        kmeanModel = KMeans(n_clusters=k, random_state=SEED).fit(X_scaled) 
        kmeanModel.fit(X_scaled)     
          
        inertias.append(kmeanModel.inertia_) 
    
    plt.plot(K, inertias, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Inertia') 
    plt.title('Elbow Method using Inertia') 
    plt.show() 
    
    kl = KneeLocator(range(2, n), inertias, curve="convex", direction="decreasing")
    elbow_point = kl.elbow
    print("Optimal number of clusters as determined by the knee locator: n_clusters =", elbow_point)
    return elbow_point



# Silhouette coefficient
def silhouette_score(X_scaled, K):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    SEED = 10
    
    silhouette_by_k=[]
    
    for n_clusters in K:
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=SEED)
        cluster_labels = clusterer.fit_predict(X_scaled)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        silhouette_by_k.append(silhouette_avg)
        
        result_silhouette_score = round(max(silhouette_by_k),5)
        result_n_clusters = K[silhouette_by_k.index(max(silhouette_by_k))]
    
    plt.plot(K, silhouette_by_k, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Silhouette') 
    plt.title('Silhouette method') 
    plt.show() 
        
    print("Max average silhouette score is:", result_silhouette_score, 
          "for n_clusters =", result_n_clusters)
    return result_n_clusters


# Gap statistic
def gap_values(X_scaled, n):
    from gap_statistic import OptimalK
    
    optimalK = OptimalK()
    
    n_clusters = optimalK(X_scaled, cluster_array=np.arange(2, n))
    
    plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
    plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
            optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')
    plt.title('Gap Values by Cluster Count')
    plt.show()
    
    print('Optimal number of clusters as determined by the gap values: n_clusters=', n_clusters)
    return n_clusters


### SVD and PCA ###

# SVD
def svd(X_scaled):
    U, D, Vt = np.linalg.svd(X_scaled)
    V = Vt.transpose()

    return U, D, Vt, V


# Finding optimal number of dimensions
# Explained variance
def explained_variance(U, D, r=10):
    from kneed import KneeLocator
    
    Z_fitted = np.matmul(U[:,0:r],np.diag(D[0:r]))
    Z_fitted.shape

    df_explained = pd.DataFrame(Z_fitted)

    plt.plot(df_explained.var())
    plt.xlabel('Dimensions')
    plt.ylabel('D')
    plt.title('Scree Plot')
    
    kl = KneeLocator(range(0, r), df_explained.var(), curve="convex", direction="decreasing")
    elbow_point = kl.elbow
   
    print("The optimal number of dimensions is:", elbow_point)
    return elbow_point
    
     
# Optimal low-rank approximation   
def low_rank_approximation(rmax, data, U, D, V, X_scaled):
    from kneed import KneeLocator
    
    errors = np.zeros(rmax-1)
    r = range(1,rmax)
    n = len(data)
    p = data.shape[1]
    for i in r:
        Zext_temp = np.matmul(U[:,0:i],np.diag(D[0:i]))
        Xapprox = np.matmul(Zext_temp,np.transpose(V[:,0:i]))
        resid = X_scaled-Xapprox
        residsq = np.matmul(resid.transpose(),resid)
        errors[i-1] = residsq.trace()
    
    #Error of low dimensional representation
    #plt.plot(r,errors/(n*p)) # errros per point
    #plt.xlabel('Final low-rank dimensions')
    #plt.ylabel('Error of low dimensional representation')
    
    #Drop in error of low dimensional representation
    plt.plot(r[1:],abs(np.diff(errors/(n*p))))
    plt.xlabel('Final low-rank dimensions')
    plt.ylabel('Drop in error of low dimensional representation')
    
    kl = KneeLocator(r[1:], abs(np.diff(errors/(n*p))), curve="convex", direction="decreasing")
    elbow_point = kl.elbow
   
    print("The optimal number of dimensions is:", elbow_point)
    
    return elbow_point