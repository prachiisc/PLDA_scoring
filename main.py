# -*- coding: utf-8 -*-

"""
Created on Sun Aug 30 20202

@author: prachi
email: prachisingh@iisc.ac.in
"""

import numpy as np
import pickle
from  matplotlib import pyplot as plt
from PLDA_scoring import PLDA_scoring
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pdb import set_trace as bp
import matplotlib as mpl

def plot_features(Xpca,Xplda,label_index):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.flat[0]
    im1 = ax.scatter(Xpca[label_index[0],0],Xpca[label_index[0],1],c='b',label='l1')
    im2 = ax.scatter(Xpca[label_index[1],0],Xpca[label_index[1],1],c='r',label='l2')
    ax.set_title('PCA transformed features')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax = axes.flat[1]
    im1 = ax.scatter(Xplda[label_index[0],0],Xplda[label_index[0],1],c='b',label='l1')
    im2 = ax.scatter(Xplda[label_index[1],0],Xplda[label_index[1],1],c='r',label='l2')
    ax.set_title('PLDA latent representations (u)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.legend([im1,im2],['Speaker1','Speaker2'])

def plot_affinity(cosine_affinity,plda_affinity):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    ax = axes
    im = ax.imshow(cosine_affinity, vmin=0, vmax=1)
    ax.set_title('Normalized Cosine affinity matrix')
    ax.set_xlabel('N')
    ax.set_ylabel('N')
    fig, axes = plt.subplots(nrows=1, ncols=1)
    ax = axes
    im = ax.imshow(plda_affinity, vmin=0, vmax=1)
    ax.set_title('Normalized PLDA affinity matrix')
    ax.set_xlabel('N')
    ax.set_ylabel('N')
    fig.colorbar(im)  
      
def plot_histogram(plda_affinity):
    plda_scores = plda_affinity[np.triu_indices(N,k=1)]
    # plt.figure()
    # plt.hist(plda_scores,rwidth=0.8)
    # plt.title('Histogran of Normalized PLDA scores')
    # plt.xlabel('PLDA scores')
    # plt.ylabel('Count')
    data = plda_scores
    nbins = 20
    minbin = data.min()
    maxbin = data.max()
   
    bins = np.linspace(minbin,maxbin,nbins)

    # cmap = plt.cm.spectral
    # cmap = plt.cm.get_cmap("nipy_spectral")
    cmap = plt.cm.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())
    colors = cmap(bins)

    hist, bin_edges = np.histogram(data, bins)

    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])
    ax1 = fig.add_axes([0.05, 0.05, 0.9, 0.1])

    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    ax.bar(bin_edges[:-1], hist, width=0.051, color=colors, alpha=0.8)
    ax.set_xlim((0., 1.))
    ax.set_title('Histogran of Normalized PLDA scores')
    ax.set_xlabel('PLDA scores')
    ax.set_ylabel('Count')

if __name__ == "__main__":
    # load features 
    xvecpath = 'xvectors/iaaa.npy'
    xvectors = np.load(xvecpath) # N X D
    xvecD = xvectors.shape[1] # Dimension of xvectors
    N = xvectors.shape[0] # number of features
    ground_labels = open('ground_labels/labels_iaaa')
    full_gndlist=[g.split()[1:] for g in ground_labels]
    gnd_list = np.array([g[0] for g in full_gndlist])
    uni_gnd_letter = np.unique(gnd_list)
    uni_gnd = np.arange(len(uni_gnd_letter))
    label_index={}
    for ind,uni in enumerate(uni_gnd_letter):
        label_index[ind] = np.where(gnd_list==uni)[0]
        gnd_list[gnd_list==uni]=ind
        
    gnd_list = gnd_list.astype(int)

    
    # load PLDA model
    pldapath = 'model/plda.pkl'
    plda = pickle.load(open(pldapath,'rb'))
    plda_obj=PLDA_scoring(plda,xvecD,target=1)
    # compute cosine similarity
    cosine_affinity = plda_obj.compute_cosine_affinity_matrix(xvectors)
    
    cosine_affinity = cosine_affinity - np.min(cosine_affinity)
    cosine_affinity = cosine_affinity/np.max(cosine_affinity)

    plda_affinity,Xpca,Xplda = plda_obj.compute_plda_affinity_matrix(plda,xvectors)
    plda_affinity = plda_affinity - np.min(plda_affinity)
    plda_affinity = plda_affinity/np.max(plda_affinity)
    
    # plot_features(Xpca,Xplda,label_index)
    # plt.show()
    plot_affinity(cosine_affinity,plda_affinity)
    plot_histogram(plda_affinity)
    plt.show()