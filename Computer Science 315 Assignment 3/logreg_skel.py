'''
March, 2014

@author: Ben
'''

from __future__ import division

import numpy as np


class logis_reg(object):
    """
    Class performing logistic regression.
    Takes as labeled data as input.
    
    Parameters
    ----------
    data : (n,d) ndarray
        An array of n, d-dimensional observations, divided into two classes
    labels : (n,) array
        Provides the label for each observation. The labels are either 0 or 1
    c : scalar (Default = 1000)
       The damping parameter, c > 0. There is less dampling for larger values 
       of c. 
      
    Attributes
    ----------
    w : (d+1,) ndarray
        The coefficients of the classifier.
        w[0] is the offset.
    d : scalar
        The dimension of the data. 
    n : scalar
        The number of observations
    data : (n,d) ndarray
           The training data
    labels : (n,) ndarray
             The labels for each observation
    c : scalar
        The damping parameter.
        Larger values mean less damping.
    
    Methods
    -------
    get_params
    sigmoid
    negloglik
    likelihood
    gradient
    hessian
    newton
    """
    def __init__(self, data, labels, c=1000):
        n,d = data.shape
        m   = len(labels)
        
        # Sanity check: Each observation should be labeled
        if not n == m:
            raise ValueError('The number of labels and observations should be the same.')
        
        bias        = np.ones((n,1))
        data        = np.hstack((bias, data))       
        self.d      = d 
        self.n      = n
        self.data   = data
        self.labels = labels
        self.c      = c
        self.w      = self.newton()
       
    def get_params(self):
        """
        Returns the parameters of the logistic regression
        
        Returns
        -------
        w : (d+1,) array
            The classification coefficients. d  equals the dimension of each 
            observation in the training data set.
            w[0] is the offset
        """
        return self.w
        
    def sigmoid(self,x):
        """
        Calculates the sigmoid function
        
        Parameters
        ----------
        x : (n,) array
            Evaluates the sigmoid function at the n scalar values.
            
        Returns
        -------
        sigmod : (n,) array
            The value of the sigmoid function at the n values of x
        """
        return 1.0/(1.0+np.exp(-x))
        
    def negloglik(self,w):
        """
        Calculate the negative log-likelihood of the training data
        for the given value of w.
        
        Paremeters
        ----------
        w : (d+1,) array
             The given classification parameters. d is the dimension 
             of each observation.
            
        Returns
        -------
        negloglik : scalar
            The negative log-likelihood
        """
        
        sig = self.sigmoid(self.data.dot(w))

        temp = self.labels*np.log(sig) + (1.-self.labels)*np.log(1.-sig)
        return -np.sum(temp) + w.dot(w)/(2.0*self.c)
        
    def likelihood(self,w):
        """
        Calculate the likelihood at w.
        
        Parameters
        ----------
        w : (d+1,) array
             The given classification parameters. Note that d equals the 
             dimension of the observations in the training set.
             
        Returns
        -------
        likelihood : scalar
            The Likelihood of the training set for given w.
            
        """
        
        return np.exp(-self.negloglik(w))
        
    def gradient(self,w):
        """
        Calculate the gradient at w.
        
        Parameters
        ----------
        w : (d+1,) array
             The given classification parameters. Note that d equals the 
             dimension of each observation in the training set.
             
        Returns
        -------
        grad : (d+1,) array
            The gradient of the negative log-likelihood, calculated at the 
            value of w
        
        """
        # Insert code here. In my implementation it is only two lines.
        negloglikw = np.array(3)
        for x in range(0,3)
            negloglikw
        grad = np.gradient(self.negloglik(w))
        return grad
    
    def hessian(self,w):
        """
        Calculate the Hessian at w
        
        Parameters
        ----------
        w : (d+1,) array
             The given classification parameters. Note that d equals the 
             dimension of the observations in the training set.
             
        Returns
        -------
        H : (d+1,d+1) ndarray
            The Hessian of the negative log-likelihood, calculated for the 
            value of w
        """  
        
        
        d = self.d
        H = np.ndarray(shape=(d+1,d+1))
        wgrad = self.gradient(w)
        for q1 in range (0,d+1):
            for q2 in range(0,d+1):
                H[q1,q2] = 1/( (wgrad[q1])*(wgrad[q2]) )
        return H
        
        

        
    def newton(self, maxiter = 10, tol = 10e-8):
        """
        Calculate the coefficients of the class separation boundary using
        Newton's method.
        
        Parameters
        ----------
        maxiter : int
            The maximum number of iterations allowed
        tol : sclar
            The desired relative accuracy
            
        Returns
        -------
        w : (d+1,) array    
            The classification parameters. Note that d equals the 
            dimension of the observations in the training set.
        
        Returns
        """
        
        err     = 1.0
        i       = 0
        w      = np.zeros((self.d+1,))
        lik     = self.likelihood(w)
        while (i < maxiter) & (err > tol):           
            update = np.linalg.solve(self.hessian(w),-self.gradient(w))
            w     += update
            lik_upd = self.likelihood(w)
            err     = np.abs(lik_upd-lik)/lik_upd
            lik     = lik_upd
            i      += 1
        return w
        
    def classify(self,x):
        """
        Classify the observations represented by x.
        
        Parameters
        ----------
        x : (d,n) ndarray
            n, d-dimensional observations that need to be classified.
            Here d is the dimension of the observations.
            
        Returns
        -------
        p : (n,) array
            The probabilities of the n observations of belonging to class with
            class label 1.
        """
        d,n = x.shape
        # Sanity check
        if d != self.d:
            raise ValueError('The dimensions of the data does not match')
            
        bias = np.ones((1,n))
        x    = np.vstack((bias,x))
        return self.sigmoid(self.w.dot(x))
          
