# Copyright 2023 <COPYRIGHT HOLDER>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is furnished 
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Math/parallelization library includes
import numpy as np
import torch

# Plotting includes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import matplotlib.tri as tri
import random
from scipy.interpolate import NearestNDInterpolator
from tqdm.notebook import tqdm
import copy
from trainingHelper import *
from sph import *

# Function that plots a set of _samples_ randomly initialized convolutions overlapped with each other
# and color mapped based on their integrals
def plotRandomWeights(samples, n = 8, basis = 'linear', windowFunction = 'Wendland2_1D', normalized = False):    
    randomWeights = []
    for i in range(samples):
        randomWeights.append(torch.rand(n) - 0.5)
    fig, axis = plt.subplots(1, 2, figsize=(16,4), sharex = False, sharey = False, squeeze = False)
    x =  torch.linspace(-1,1,511)

    fx = evalBasisFunction(n, x , which = basis, periodic=False)
    fx = fx / torch.sum(fx, axis = 0)[None,:] if normalized else fx # normalization step
    windowFn = getWindowFunction(windowFunction) # window function that is applied after each network layer

    integrals = []
    for i in range(len(randomWeights)):
        integral = torch.sum(torch.sum(randomWeights[i][:,None] * fx,axis=0)) * 2 / 511
        integrals.append(integral)
    integrals = torch.hstack(integrals)
    norm = mpl.colors.Normalize(vmin=torch.min(integrals), vmax=torch.max(integrals))

    for i in range(len(randomWeights)):
        axis[0,0].plot(x,torch.sum(randomWeights[i][:,None] * fx,axis=0),ls='-',c=cmap(norm(integrals[i])), label = '$\Sigma_i w_i f_i(x)$', alpha = 0.75)
        axis[0,1].plot(x,windowFn(torch.abs(x)) * torch.sum(randomWeights[i][:,None] * fx,axis=0),ls='-',c=cmap(norm(integrals[i])), label = '$\Sigma_i w_i f_i(x)$', alpha = 0.75)
    axis[0,0].set_title('Random initializations %s [%2d]'% (basis,n))
    axis[0,1].set_title('Random initializations %s [%2d] /w window'% (basis,n))

    fig.tight_layout()

# Function that plots a set of weights (given by a state dict) for visualization of the convolution of a network
def plotWeights(dict, basis, normalized):
    fig, axis = plt.subplots(1, 2, figsize=(16,4), sharex = False, sharey = False, squeeze = False)
    x =  torch.linspace(-1,1,511)
    n = dict['weight'].shape[0]
    fx = evalBasisFunction(n, x , which = basis, periodic=False)
    fx = fx / torch.sum(fx, axis = 0)[None,:] if normalized else fx # normalization step
    # plot the individual basis functions with a weight of 1
    for y in range(n):
        axis[0,0].plot(x, fx[y,:], label = '$f_%d(x)$' % y)
    # plot the overall convolution basis for all weights equal to 1
    axis[0,0].plot(x,torch.sum(fx, axis=0),ls='--',c='white', label = '$\Sigma_i f_i(x)$')
    axis[0,0].set_title('Basis Functions')

    axis[0,1].plot(x,torch.sum(dict['weight'][:,0].detach() * fx,axis=0) + dict['bias'].detach(),ls='--',c='white', label = '$\Sigma_i w_i f_i(x)$')
    axis[0,1].set_title('Learned convolution')

    fig.tight_layout()

# Plots the given simulation (via simulationStates) and the given timesteps
# This function plots both density and velocity
def plotSimulationState(simulationStates, minDomain, maxDomain, dt, timepoints = []):
    fig, axis = plt.subplots(2, 1, figsize=(9,6), sharex = True, sharey = False, squeeze = False)

    axis[0,0].axvline(minDomain, color = 'black', ls = '--')
    axis[0,0].axvline(maxDomain, color = 'black', ls = '--')
    axis[1,0].axvline(minDomain, color = 'black', ls = '--')
    axis[1,0].axvline(maxDomain, color = 'black', ls = '--')

    axis[1,0].set_xlabel('Position')
    axis[1,0].set_ylabel('Velocity[m/s]')
    axis[0,0].set_ylabel('Density[1/m]')

    def plotTimePoint(i, c, simulationStates, axis):
        x = simulationStates[i,0,:]
        y = simulationStates[i,c,:]
        idx = torch.argsort(x)
        axis.plot(x[idx].detach().cpu().numpy(), y[idx].detach().cpu().numpy(), label = 't = %1.2g' % (i * dt))
    if timepoints == []:
        plotTimePoint(0,1, simulationStates, axis[1,0])
        plotTimePoint(0,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]//4,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]//4,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]//4*2,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]//4*2,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]//4*3,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]//4*3,2, simulationStates, axis[0,0])
        
        plotTimePoint(simulationStates.shape[0]-1,1, simulationStates, axis[1,0])
        plotTimePoint(simulationStates.shape[0]-1,2, simulationStates, axis[0,0])
    else:
        for t in timepoints:
            plotTimePoint(t,1, simulationStates, axis[1,0])
            plotTimePoint(t,2, simulationStates, axis[0,0])

    axis[0,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axis[1,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

# Plots the density of a set of particles, as determined using SPH, as well as the FFT and PSD of the density field
# Used for plotting the initial density field/sampling
def plotDensityField(fluidPositions, fluidAreas, minDomain, maxDomain, particleSupport):
    ghostPositions = createGhostParticles(fluidPositions, minDomain, maxDomain)
    fluidNeighbors, fluidRadialDistances, fluidDistances = findNeighborhoods(fluidPositions, ghostPositions, particleSupport)
    fluidDensity = computeDensity(fluidPositions, fluidAreas, particleSupport, fluidRadialDistances, fluidNeighbors)

    xs = fluidPositions.detach().cpu().numpy()
    densityField = fluidDensity.detach().cpu().numpy()
    fig, axis = plt.subplots(1, 3, figsize=(18,6), sharex = False, sharey = False, squeeze = False)
    numSamples = densityField.shape[-1]
    fs = numSamples/2
    fftfreq = np.fft.fftshift(np.fft.fftfreq(xs.shape[-1], 1/fs/1))    
    x = densityField
    y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
    axis[0,0].plot(xs, densityField)
    axis[0,1].loglog(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = 'baseTarget')
    f, Pxx_den = scipy.signal.welch(densityField, fs, nperseg=len(x)//32)
    axis[0,2].loglog(f, Pxx_den, label = 'baseTarget')
    axis[0,2].set_xlabel('frequency [Hz]')
    axis[0,2].set_ylabel('PSD [V**2/Hz]')
    fig.tight_layout()
    return fluidDensity
# Plots a pseudo 2D plot of the given simulation showing both the density and velocity (color mapped) with
# time on the y-axis and position on the x-axis to demonstrate how the simulations evolve over time.
# This function is relatively slow as it does a resampling from the particle data to a regular grid, of size
# nx * ny, using NearestNDInterpolator.
def regularPlot(simulationStates, minDomain, maxDomain, dt, nx = 512, ny = 2048):
    timeArray = torch.arange(simulationStates.shape[0])[:,None].repeat(1,simulationStates.shape[2]) * dt
    positionArray = simulationStates[:,0]
    xys = torch.vstack((timeArray.flatten().to(positionArray.device).type(positionArray.dtype), positionArray.flatten())).mT.detach().cpu().numpy()

    interpVelocity = NearestNDInterpolator(xys, simulationStates[:,1].flatten().detach().cpu().numpy())
    interpDensity = NearestNDInterpolator(xys, simulationStates[:,2].flatten().detach().cpu().numpy())

    X = torch.linspace(torch.min(timeArray), torch.max(timeArray), ny).detach().cpu().numpy()
    Y = torch.linspace(torch.min(positionArray), torch.max(positionArray), nx).detach().cpu().numpy()
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    # Z = interp(X, Y)

    fig, axis = plt.subplots(2, 1, figsize=(14,9), sharex = False, sharey = False, squeeze = False)


    im = axis[0,0].pcolormesh(X,Y,interpDensity(X,Y), cmap = 'viridis', vmin = torch.min(torch.abs(simulationStates[:,2])),vmax = torch.max(torch.abs(simulationStates[:,2])))
    # im = axis[0,0].imshow(simulationStates[:,2].mT, extent = [0,dt * simulationStates.shape[0], maxDomain,minDomain])
    axis[0,0].set_aspect('auto')
    axis[0,0].set_xlabel('time[/s]')
    axis[0,0].set_ylabel('position')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[0,0].axhline(minDomain, color = 'black', ls = '--')
    axis[0,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Density [1/m]')

    im = axis[1,0].pcolormesh(X,Y,interpVelocity(X,Y), cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    # im = axis[1,0].imshow(simulationStates[:,1].mT, extent = [0,dt * simulationStates.shape[0], maxDomain,minDomain], cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    axis[1,0].set_aspect('auto')
    axis[1,0].set_xlabel('time[/s]')
    axis[1,0].set_ylabel('position')
    ax1_divider = make_axes_locatable(axis[1,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[1,0].axhline(minDomain, color = 'black', ls = '--')
    axis[1,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Velocity [m/s]')

    fig.tight_layout()

# This function is used as part of the functionality to plot the networks performance over the entire simulation.
# This function in particular valuates the prediction, and corresponding loss, of the given network (and weights)
# for the given set of timesteps in bdata.
def computeEvaluationLoss(model, weights, bdata, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked, batchSize = 128):  
    batched = np.array_split(bdata, len(bdata) // batchSize + 1)
    predictions = []
    groundTruths = []
    lossTerms = []
    losses = []     
    with torch.no_grad():
        storedWeights = copy.deepcopy(model.state_dict())
        model.load_state_dict(weights)
    for batch in tqdm(batched, leave = False):        
        with torch.no_grad():
            stackedPositions, features, groundTruth, stackedNeighbors, d = loadBatch(simulationStates, minDomain, maxDomain, particleSupport, batch, getFeatures, getGroundTruth, stacked)
            prediction = model((features[:,None], features[:,None]), stackedNeighbors, d)
            lossTerm = lossFunction(prediction, groundTruth)
            loss = torch.mean(lossTerm)
            predictions.append(prediction)
            groundTruths.append(groundTruth)
            lossTerms.append(lossTerm)
            losses.append(loss)  
    with torch.no_grad():
            model.load_state_dict(storedWeights)
    return torch.cat(predictions, axis = 0).cpu(), torch.cat(groundTruths, axis = 0).cpu(), torch.cat(lossTerms, axis = 0).cpu(), torch.hstack(losses).cpu()

# Helper function to plot 2 2D datasets side by side with linear color mapping
def plotAB(fig, axisA, axisB, dataA, dataB, batchesA, batchesB, numParticles, cmap = 'viridis'):
    vmin = min(torch.min(dataA), torch.min(dataB))
    vmax = max(torch.max(dataA), torch.max(dataB))
    imA = axisA.imshow(dataA.reshape((batchesA.shape[0], numParticles)).mT, cmap = cmap, interpolation = 'nearest', vmin = vmin, vmax = vmax, extent = [np.min(batchesA),np.max(batchesA),numParticles,0]) # uses some matrix reshaping to undo the hstack
    imB = axisB.imshow(dataB.reshape((batchesB.shape[0], numParticles)).mT, cmap = cmap, interpolation = 'nearest', vmin = vmin, vmax = vmax, extent = [np.min(batchesB),np.max(batchesB),numParticles,0]) # uses some matrix reshaping to undo the hstack
    axisA.axis('auto')
    axisB.axis('auto')
    ax1_divider = make_axes_locatable(axisB)
    cax1 = ax1_divider.append_axes("right", size="5%", pad="2%")
    cbarPredFFT = fig.colorbar(imB, cax=cax1,orientation='vertical')
    cbarPredFFT.ax.tick_params(labelsize=8) 
# Helper function to plot 2 2D datasets side by side with logarithmic color mapping
def plotABLog(fig, axisA, axisB, dataA, dataB, batchesA, batchesB, numParticles, cmap = 'viridis'):
    vmin = min(np.percentile(dataA[dataA > 0], 1), np.percentile(dataB[dataB > 0],1))
    vmax = max(torch.max(dataA), torch.max(dataB))
    imA = axisA.imshow(dataA.reshape((batchesA.shape[0], numParticles)).mT, cmap = cmap, interpolation = 'nearest', norm = LogNorm(vmin=vmin, vmax=vmax), extent = [np.min(batchesA),np.max(batchesA),numParticles,0]) # uses some matrix reshaping to undo the hstack
    imB = axisB.imshow(dataB.reshape((batchesB.shape[0], numParticles)).mT, cmap = cmap, interpolation = 'nearest', norm = LogNorm(vmin=vmin, vmax=vmax), extent = [np.min(batchesB),np.max(batchesB),numParticles,0]) # uses some matrix reshaping to undo the hstack
    axisA.axis('auto')
    axisB.axis('auto')
    ax1_divider = make_axes_locatable(axisB)
    cax1 = ax1_divider.append_axes("right", size="5%", pad="2%")
    cbarPredFFT = fig.colorbar(imB, cax=cax1,orientation='vertical')
    cbarPredFFT.ax.tick_params(labelsize=8) 
# Plots the learning progress of a training progress, see the main jupyter notebook for an example on how to use this function
def plotAll(model, device, weights, basis, normalized, iterations, epochs, numParticles, batchSize, lossArray, simulationStates, minDomain, maxDomain, particleSupport, timestamps, testBatch, lossFunction, getFeatures, getGroundTruth, stacked):
    trainingPrediction, trainingGroundTruth, trainingLossTerm, trainingLoss = computeEvaluationLoss(model, weights[-1][-1], timestamps, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
    testingPrediction, testingGroundTruth, testingLossTerm, testingLoss = computeEvaluationLoss(model, weights[-1][-1], testBatch, lossFunction, simulationStates, minDomain, maxDomain, particleSupport, getFeatures, getGroundTruth, stacked)
    fig, axis = plt.subplot_mosaic('''AABB
    AABB
    CCCD
    EEEF
    GGGH''', figsize=(16,10), sharex = False, sharey = False)
    fig.suptitle('Training results for basis %s%s %2d epochs %4d iterations batchSize %d: %2.6g' % (basis, '' if not normalized else ' (normalized)', epochs, iterations, batchSize, np.mean(lossArray[-1][-1])))

    batchedLosses = np.stack(lossArray, axis = 0).reshape(iterations * epochs, numParticles * batchSize)
    axis['A'].set_title('Learning progress')
    axis['A'].semilogy(np.mean(batchedLosses, axis = 1))
    axis['A'].semilogy(np.min(batchedLosses, axis = 1))
    axis['A'].semilogy(np.max(batchedLosses, axis = 1))

    axis['C'].set_title('Prediction (Training)') 
    axis['D'].set_title('Prediction (Testing)')
    axis['E'].set_title('Ground Truth (Training)') 
    axis['F'].set_title('Ground Truth (Testing)')
    axis['G'].set_title('Loss (Training)') 
    axis['H'].set_title('Loss (Testing)') 
    plotAB(fig, axis['C'], axis['D'], trainingPrediction, testingPrediction, timestamps, testBatch, numParticles, cmap = 'viridis')
    plotAB(fig, axis['E'], axis['F'], trainingGroundTruth, testingGroundTruth, timestamps, testBatch, numParticles, cmap = 'viridis')
    plotABLog(fig, axis['G'], axis['H'], trainingLossTerm, testingLossTerm, timestamps, testBatch, numParticles, cmap = 'viridis')
    axis['C'].set_xticklabels([])
    axis['D'].set_xticklabels([])
    axis['E'].set_xticklabels([])
    axis['F'].set_xticklabels([])
    axis['D'].set_yticklabels([])
    axis['F'].set_yticklabels([])
    axis['H'].set_yticklabels([])
    
    
    cm = mpl.colormaps['viridis']

    x =  torch.linspace(-1,1,511)[:,None].to(device)
    fx = torch.ones(511)[:,None].to(device)
    neighbors = torch.vstack((torch.zeros(511).type(torch.long), torch.arange(511).type(torch.long)))
    neighbors = torch.vstack((torch.arange(511).type(torch.long), torch.zeros(511).type(torch.long))).to(device)
    steps = iterations * epochs
    ls = np.logspace(0, np.log10(steps), num =  50)
    ls = [int(np.floor(f)) for f in ls]
    ls = np.unique(ls).tolist()

    storedWeights = copy.deepcopy(model.state_dict())
    c = 0
    for i in tqdm(range(epochs), leave = False):
        for j in tqdm(range(iterations), leave = False):
            c = c + 1        
            if c + 1 in ls:           
                
                model.load_state_dict({k: v.to(device) for k, v in weights[i][j].items()})
                axis['B'].plot(x[:,0].detach().cpu().numpy(), model((fx,fx), neighbors, x).detach().cpu().numpy(),ls='--',c= cm(ls.index(c+1) / (len(ls) - 1)), alpha = 0.95)

    model.load_state_dict(storedWeights)
    axis['B'].set_title('Filter progress')

    fig.tight_layout()
    model.load_state_dict({k: v.to(device) for k, v in weights[-1][-1].items()})

    return fig, axis

