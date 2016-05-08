# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:10:38 2016

@author: rob
"""

import matplotlib.pyplot as plt
import numpy as np

def testfunction(x):
    return x**2


def plot_heat(M,O):
    """
    Function takes in a 4D np array with dimenstion
    -1 Number of samples
    -2 Width of heatmap
    -3 Height of heatmap
    -4 Number of channels
    and takes in the original image as 2D np array with dimension
    -1 Number of samples
    -2 Width*height of image
    
    Plots a collections of heatmaps
    """
    assert len(M.shape) == 4,'We expect a 4D np array'
    N,H,W,C = M.shape
    assert C <= 11, 'For now, the function only works with less than 11 channels'
    assert O.shape[0] == N, 'The original contains more samples than the provided 4D array'
    w_im = np.sqrt(O.shape[1])
    assert w_im.is_integer(),'For now, the plot_heat() expects a square original'
    w_im = int(w_im)
    
    
    print('Print 1 of %s samples, shape %s by %s with %s channels' %(N,H,W,C))
    
    ind = np.random.choice(N)
    
    #Min and maxvalue are used to set similar color scales for all heatmaps
    minvalue = np.min(M[ind,:,:,:])
    maxvalue = np.max(M[ind,:,:,:])

    if True:  #Set true
        f, axarr = plt.subplots(4, 3)  #4 and 3 are hardcoded to plot 11 heatmaps and 1 true image
        for i in xrange(12):
            (x,y) = np.unravel_index(i,(4,3))
            if i == 0:
                im = np.reshape(O[ind,:],(w_im,w_im))
                axarr[x,y].imshow(im.T)
            else:
                #plot heatmaps
                if i <=10: #For our case we only have 10 plots to make. Change this
                # if you are working with different datasets
                    axarr[x,y].imshow(M[ind,:,:,i-1].T,origin='upper',vmin=minvalue,vmax=maxvalue,interpolation='none')
            plt.setp([axarr[x,y].get_xticklabels()], visible=False)
            plt.setp([axarr[x,y].get_yticklabels()], visible=False)
            f.subplots_adjust(hspace=0)  #No horizontal space between subplots
            f.subplots_adjust(wspace=0)  #No vertical space between subplots    