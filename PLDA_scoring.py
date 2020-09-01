# -*- coding: utf-8 -*-

"""
Created on Sun Aug 30 20202

@author: prachi
email: prachisingh@iisc.ac.in
"""

import numpy as np
import math
# value given in https://kaldi-asr.org/doc/kaldi-math_8h_source.html
M_LOG_2PI = 1.8378770664093454835606594728112

class PLDA_scoring():
    def __init__(self,plda,dimension=128,pca_dimension=10,target=1):
        
        self.plda = plda
        self.mean_vec = plda['mean_vec']
        self.transform_mat = plda['transform_mat']
        self.target_energy = 0.1
        self.pca_dimension = pca_dimension
        self.target = target

    def preprocessing(self,X):
        """
        Perform mean_subtraction using mean.vec-> apply transform.mat -> input length norm

        Parameters
        ----------
        X : Xvectors N X d

        Returns
        -------
        transformed xvectors N X D

        """
        
        xvecs = X.T # DX N
        dim = xvecs.shape[1]
        # preprocessing
        # mean subtraction
        xvecs = xvecs - self.mean_vec[:,np.newaxis]
        # PCA transform
        xvecs = self.transform_mat @ xvecs
        l2_norm = np.linalg.norm(xvecs, axis=0, keepdims=True)
        l2_norm = l2_norm/math.sqrt(dim)

        xvecsnew = xvecs/l2_norm
        
        return xvecsnew.T

    def compute_PCA_transform(self,X):
        """
        Computes filewise PCA
        given in https://kaldi-asr.org/doc/ivector-plda-scoring-dense_8cc_source.html
        Apply transform on mean shifted xvectors

        Parameters
        ----------
        Xvectors

        Returns
        ----------
        new xvectors and transform

        """
    
        xvec = X #N X D
        num_rows = xvec.shape[0]
        num_cols = xvec.shape[1]
        mean = np.mean(xvec,0,keepdims=True)
        S = np.matmul(xvec.T,xvec)
        S = S/num_rows
        
        S = S - mean.T @ mean
   
        try:           
            ev_s, eig_s , _ = np.linalg.svd(S,full_matrices=True)
        except:
            print('SVD_error')
        if not self.target:
            dim = self.pca_dimension
        else:
            total_energy = np.sum(eig_s)
            
            energy =0.0
            dim=1
            while energy/total_energy <= self.target_energy:
                energy += eig_s[dim-1]
                dim +=1
            print('pca_dim computed: ',dim)
        transform = ev_s[:,:dim]
      
        transxvec = xvec @ transform
        newX = transxvec
      
        return newX, transform.T


    def applytransform_plda(self,transform_in):
        """
        Apply PCA filewise transform on PLDA parameters
        details are given in : https://kaldi-asr.org/doc/classkaldi_1_1Plda.html#afda9c0178f439b40698914f237adef81

        Parameters
        ----------
        transform_in : numpy  D X dim
           PCA filewise transform

        """
        
        mean_plda = self.plda['plda_mean']
        #transfomed mean vector
        new_mean = transform_in @ mean_plda[:,np.newaxis]
        D = self.plda['diagonalizing_transform']
        psi = self.plda['Psi_across_covar_diag']
        D_inv = np.linalg.inv(D)
        # within class and between class covarinace
        phi_b=  (D_inv * psi.reshape(1,-1)) @ D_inv.T
        phi_w = D_inv @ D_inv.T
        # transformed with class and between class covariance
        new_phi_b = transform_in @ phi_b @ transform_in.T
        new_phi_w = transform_in @ phi_w @ transform_in.T
        ev_w, eig_w,_ =np.linalg.svd(new_phi_w)
        eig_w_inv = 1/np.sqrt(eig_w)
        Dnew = eig_w_inv.reshape(-1,1)*ev_w.T
        new_phi_b_proj = Dnew @ new_phi_b @ Dnew.T
        ev_b, eig_b,_ = np.linalg.svd(new_phi_b_proj)
        psi_new = eig_b

        Dnew = ev_b.T @ Dnew
        self.plda['plda_mean'] = new_mean
        self.plda['diagonalizing_transform'] = Dnew
        self.plda['Psi_across_covar_diag'] = psi_new
        self.plda['offset'] = -Dnew @ new_mean.reshape(-1,1)
        # ac = res['Psi_across_covar_diag']
        tot = 1 + psi_new
        self.plda['diagP'] = psi_new/(tot*(tot-psi_new*psi_new/tot))
        self.plda['diagQ'] = (1/tot) - 1/(tot - psi_new*psi_new/tot)
        
    def transformXvectors(self,X):
        """
        Apply plda mean and diagonalizing transform to xvectors for scoring

        Parameters
        ----------
        X : TYPE
           Xvectors 1 X N X D

        Returns
        -------
        X_new : TYPE
            transformed x-vectors

        """
        
        offset = self.plda['offset']
        offset = offset.T
     
        D = self.plda['diagonalizing_transform']
        Dnew = D.T
        X_new = X @ Dnew
        X_new = X_new + offset
        # Get normalizing factor
        # Defaults : normalize_length(true), simple_length_norm(false)
        X_new_sq = X_new**2
        psi = self.plda['Psi_across_covar_diag']
        inv_covar = (1.0/(1.0+psi)).reshape(-1,1)
        dot_prod = X_new_sq @ inv_covar # N X 1
        Dim = D.shape[0]
        normfactor = np.sqrt(Dim/dot_prod)
        X_new = X_new*normfactor
        
        return X_new
    
    def compute_plda_score(self,X):
        """
        Computes plda affinity matrix using Loglikelihood function

        Parameters
        ----------
        X : TYPE
            X-vectors 1 X N X D

        Returns
        -------
        Affinity matrix TYPE
            1 X N X N 

        """
        
        psi = self.plda['Psi_across_covar_diag']
        mean = psi/(psi+1.0)
        mean = mean.reshape(1,-1)*X # N X D , X[0]- Train xvectors
        
        # given class computation
        variance_given = 1.0 + psi/(psi+1.0)
        logdet_given = np.sum(np.log(variance_given))
        variance_given = 1.0/variance_given
        
        # without class computation
        variance_without =1.0 + psi
        logdet_without = np.sum(np.log(variance_without))
        variance_without = 1.0/variance_without
        
        sqdiff = X #---- Test x-vectors
        nframe = X.shape[0]
        dim = X.shape[1]
        loglike_given_class = np.zeros((nframe,nframe))
        for i in range(nframe):
            sqdiff_given = sqdiff - mean[i]
            sqdiff_given  =  sqdiff_given**2
            
            loglike_given_class[:,i] = -0.5 * (logdet_given + M_LOG_2PI * dim + \
                                   np.matmul(sqdiff_given, variance_given))
        sqdiff_without = sqdiff**2
        loglike_without_class = -0.5 * (logdet_without + M_LOG_2PI * dim + \
                                     np.matmul(sqdiff_without, variance_without))
        loglike_without_class = loglike_without_class.reshape(-1,1) 
        # loglike_given_class - N X N, loglike_without_class - N X1
        loglike_ratio = loglike_given_class - loglike_without_class  # N X N
        
        return loglike_ratio
          
    def compute_plda_affinity_matrix(self,plda,X):
        """Compute the plda_affinity matrix from data.
        plda functions given in https://kaldi-asr.org/doc/classkaldi_1_1Plda.html#afda9c0178f439b40698914f237adef81
        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            affinity: numpy array of shape (n_samples, n_samples)
        """
        nframe = X.shape[0]
        self.plda = plda.copy()
        
        X = self.preprocessing(X) #output -N X D
        
        X, PCA_transform = self.compute_PCA_transform(X)
        Xpca = X.copy()
        self.applytransform_plda(PCA_transform)
        X = self.transformXvectors(X)
        Xplda = X.copy() 
        affinity = self.compute_plda_score(X)
       
        return affinity,Xpca,Xplda

    def compute_cosine_affinity_matrix(self,X):
        """Compute the cosine affinity matrix from data.

        Note that the range of affinity is [-1,1].

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            affinity: numpy array of shape (n_samples, n_samples)
        """
        # Normalize the data.
        l2_norms = np.linalg.norm(X, axis=1,keepdims=True)
        X_normalized = X / l2_norms
        # Compute cosine similarities. Range is [-1,1].

        cosine_similarities = X_normalized @ X_normalized.T
        # Compute the affinity. Range is [0,1].
        # Note that this step is not mentioned in the paper!
        # affinity = cosine_similarities
        affinity = cosine_similarities
        # affinity = (cosine_similarities + 1.0) / 2.0

        return affinity

   

