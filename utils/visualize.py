import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines
import sys
import os
import csv

from .misc import *   

__all__ = ['make_image', 'show_batch', 'show_mask', 'show_mask_single', 'plot_results']

# functions to show an image
def make_image(img, mean=(0,0,0), std=(1,1,1)):
    for i in range(0, 3):
        img[i] = img[i] * std[i] + mean[i]    # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def gauss(x,a,b,c):
    return torch.exp(-torch.pow(torch.add(x,-b),2).div(2*c*c)).mul(a)

def colorize(x):
    ''' Converts a one-channel grayscale image to a color heatmap image '''
    if x.dim() == 2:
        torch.unsqueeze(x, 0, out=x)
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[1] = gauss(x,1,.5,.3)
        cl[2] = gauss(x,1,.2,.3)
        cl[cl.gt(1)] = 1
    elif x.dim() == 4:
        cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
        cl[:,0,:,:] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[:,1,:,:] = gauss(x,1,.5,.3)
        cl[:,2,:,:] = gauss(x,1,.2,.3)
    return cl

def show_batch(images, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.imshow(images)
    plt.show()


def show_mask_single(images, mask, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * Std[i] + Mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.subplot(2, 1, 1)
    plt.imshow(images)
    plt.axis('off')

    # for b in range(mask.size(0)):
    #     mask[b] = (mask[b] - mask[b].min())/(mask[b].max() - mask[b].min())
    mask_size = mask.size(2)
    # print('Max %f Min %f' % (mask.max(), mask.min()))
    mask = (upsampling(mask, scale_factor=im_size/mask_size))
    # mask = colorize(upsampling(mask, scale_factor=im_size/mask_size))
    # for c in range(3):
    #     mask[:,c,:,:] = (mask[:,c,:,:] - Mean[c])/Std[c]

    # print(mask.size())
    mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
    # mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask), Mean, Std)
    plt.subplot(2, 1, 2)
    plt.imshow(mask)
    plt.axis('off')

def show_mask(images, masklist, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * Std[i] + Mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.subplot(1+len(masklist), 1, 1)
    plt.imshow(images)
    plt.axis('off')

    for i in range(len(masklist)):
        mask = masklist[i].data.cpu()
        # for b in range(mask.size(0)):
        #     mask[b] = (mask[b] - mask[b].min())/(mask[b].max() - mask[b].min())
        mask_size = mask.size(2)
        # print('Max %f Min %f' % (mask.max(), mask.min()))
        mask = (upsampling(mask, scale_factor=im_size/mask_size))
        # mask = colorize(upsampling(mask, scale_factor=im_size/mask_size))
        # for c in range(3):
        #     mask[:,c,:,:] = (mask[:,c,:,:] - Mean[c])/Std[c]

        # print(mask.size())
        mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
        # mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask), Mean, Std)
        plt.subplot(1+len(masklist), 1, i+2)
        plt.imshow(mask)
        plt.axis('off')


def get_xydata(header, rows, xcol, ycol, label=None, ylim=None):
    xdata, ydata = [], []
    xidx = None if xcol is None else header[xcol]
    yidx = None if ycol is None else header[ycol]
    for i, row in enumerate(rows):
        xdata.append(i if xidx is None else float(row[xidx]))
        ydata.append(i if yidx is None else float(row[yidx]))
    return xdata, ydata, label or ycol, ylim

def get_plot_data(file_xy_name_label_lims):
    xy_data = []
    for t in file_xy_name_label_lims:
        log_file, xcol, ycol, ylabel, ylim = t
        header = {}
        rows = []
        with open(log_file, mode='r') as lf:
            reader = csv.reader(lf, delimiter='\t')
            header_row = next(reader)
            header = dict(list((j, i) for i,j in enumerate(header_row)))
            rows = list(reader)
            xy_data.append(get_xydata(header, rows, xcol, ycol, ylabel, ylim))
    return xy_data

def plot_log(title, xlabel, xy_data, xlim=None, cm_name='Dark2', cm_count=8):
    figure = plt.figure(figsize=(10, 5))
    cm = plt.get_cmap(cm_name)
    ax1 = figure.add_subplot(111)
    ax1.grid(True)
    ax1.set_xlabel(xlabel)
    ax1.xaxis.label.set_style('italic')
    if xlim is not None:
       ax1.set_xlim(xlim) 
    ax1.spines['right'].set_color((.8,.8,.8))
    ax1.spines['top'].set_color((.8,.8,.8))
    ax_title = ax1.set_title(title)
    ax_title.set_weight('bold')

    for i, (xdata, ydata, ylabel, ylim) in enumerate(xy_data):
        if i == 0:
            ax = ax1
        else:
            ax = ax1.twinx()
        
        color = cm(i / float(len(xy_data)))
        line = matplotlib.lines.Line2D(xdata, ydata, label=ylabel, color=color)
        ax.add_line(line)
        
        ax.set_ylabel(ylabel)
        ax.yaxis.label.set_color(color)
        ax.yaxis.label.set_style('italic')
        if ylim is not None:
            ax.set_ylim(ylim) 
        if i > 0:
            pos = i * 50
            ax.spines['right'].set_position(('outward', pos))
            
        ax.relim()
        ax.autoscale_view()
    figure.legend(loc='lower right')
    figure.tight_layout()

def plot_results(exp_names, dataset_name, nw_name, title=None, xlabel='Epoch', results_base_dir='results', file_name='log.txt'):
    xy_name_label_lim = []
    
    file_xy_name_label_lims = []
    for exp_name in exp_names:
        log_file = log_dir = os.path.join(results_base_dir, exp_name, dataset_name, nw_name, file_name)
        file_xy_name_label_lims.append((log_file, None, 'Train Loss', 'Train Loss-' + exp_name, None)) 
        file_xy_name_label_lims.append((log_file, None, 'Valid Acc.', 'Test Acc-' + exp_name, (0, 100)))
    
    
    plot_data = get_plot_data(file_xy_name_label_lims)
    plot_log(title or (dataset_name + '-' + nw_name), xlabel, plot_data)



