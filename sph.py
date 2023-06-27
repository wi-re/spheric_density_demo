# Copyright 2023 Rene Winchenbach
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

# Imports for neighborhood searches later on
from torch_geometric.nn import radius
from cutlass import scatter_sum

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

# Wendland 2 kernel as per Dehnen and Aly 2012 https://arxiv.org/abs/1204.2471
C = 5/4
def kernel(q, support):
    return C * (1-q)**3 * (1 + 3 * q) / (support)

def kernelGradient(q, dist, support):
    return -dist * C * 12 * (q-1)**2 * q / (support ** 2)

# Ghost particle creation for periodic BC, these particles not actually used
# but only used as part of the neighborhood search
def createGhostParticles(particles, minDomain, maxDomain):
    ghostParticlesLeft = particles - (maxDomain - minDomain)
    ghostParticlesRight = particles + (maxDomain - minDomain)

    allParticles = torch.hstack((ghostParticlesLeft, particles, ghostParticlesRight))
    return allParticles

# Neighborhood search
def findNeighborhoods(particles, allParticles, support):
    # Call the external neighborhood search function
    row, col = radius(allParticles, particles, support, max_num_neighbors = 256)
    fluidNeighbors = torch.stack([row, col], dim = 0)
        
    # Compute the distances of all particle pairings
    fluidDistances = (allParticles[fluidNeighbors[1]] - particles[fluidNeighbors[0]])
    # This could also be done with an absolute value function
    fluidRadialDistances = torch.abs(fluidDistances)# torch.sqrt(fluidDistances**2)

    # Compute the direction, in 1D this is either 0 (i == j) or +-1 depending on the relative position
    fluidDistances[fluidRadialDistances < 1e-7] = 0
    fluidDistances[fluidRadialDistances >= 1e-7] /= fluidRadialDistances[fluidRadialDistances >= 1e-7]
    fluidRadialDistances /= support
    
    # Modify the neighbor list so that everything points to the original particles
    particleIndices = torch.arange(particles.shape[0]).to(particles.device)
    stackedIndices = torch.hstack((particleIndices, particleIndices, particleIndices))
    fluidNeighbors[1,:] = stackedIndices[fluidNeighbors[1,:]]    
    
    return fluidNeighbors, fluidRadialDistances, fluidDistances

# Summation density formulation. Note that we ignore the rest density here as this term cancels out everywhere
# that we use it and it makes learning this term more straight forward.
def computeDensity(particles, particleArea, particleSupport, fluidRadialDistances, fluidNeighbors):
    pairWiseDensity = particleArea[fluidNeighbors[1]] * kernel(fluidRadialDistances, particleSupport)
    fluidDensity = scatter_sum(pairWiseDensity, fluidNeighbors[0], dim=0, dim_size = particles.shape[0])
    
    return fluidDensity

# symmetric pressure gradient based on DJ Price 2010 https://arxiv.org/pdf/1012.1885.pdf
def computePressureForces(fluidPositions, fluidDensity, fluidPressure, fluidAreas, particleSupport, restDensity, fluidNeighbors, fluidRadialDistances, fluidDistances):
    i = fluidNeighbors[0,:]
    j = fluidNeighbors[1,:]

    pairwisePressureForces = fluidAreas[j] * restDensity * \
            (fluidPressure[i] / (fluidDensity[i] * restDensity)**2 + fluidPressure[j] / (fluidDensity[j]* restDensity)**2) *\
            kernelGradient(fluidRadialDistances, fluidDistances, particleSupport)
    fluidPressureForces = scatter_sum(pairwisePressureForces, fluidNeighbors[0], dim=0, dim_size = fluidPositions.shape[0])
    
    return fluidPressureForces

# Laminar viscosity term based on DJ Price 2010 https://arxiv.org/pdf/1012.1885.pdf
def computeLaminarViscosity(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, rhoi, rhoj, ui, uj, alpha : float, beta: float, c0 : float, restDensity : float):
    gradW = kernelGradient(radialDistances, distances, support)

    uij = ui[i] - uj[j]
    rij = ri[i] - rj[j]
    rij = -distances * radialDistances
    
    mu_nom = support * (uij * rij)
    mu_denom = torch.abs(rij) + 0.01 * support**2
    mu = mu_nom / mu_denom
    
    
    nom = - alpha * c0 * mu + beta * mu**2
    denom = (rhoi[i] + rhoj[j]) / 2
    termL = Vi[j] * nom / denom
    
    term = termL * gradW
    return scatter_sum(term, i, dim=0, dim_size = Vi.shape[0])
# Helper function that calls the laminar viscosity function
def computeDiffusion(fluidPositions, fluidVelocities, fluidAreas, fluidDensities, particleSupport, restDensity, alpha, beta, c0, fluidNeighbors, fluidRadialDistances, fluidDistances):
    laminarViscosity = computeLaminarViscosity(fluidNeighbors[0], fluidNeighbors[1], \
                                                                                      fluidPositions, fluidPositions, fluidAreas, fluidAreas,\
                                                                                      fluidDistances, fluidRadialDistances,\
                                                                                      particleSupport, fluidDensities.shape[0], 1e-7,\
                                                                                      fluidDensities, fluidDensities,\
                                                                                      fluidVelocities,fluidVelocities,
                                                                                      alpha, beta, c0, restDensity)
    
    return laminarViscosity
# XSPH based numerical viscosity term based on Monaghan 2005 https://ui.adsabs.harvard.edu/link_gateway/2005RPPh...68.1703M/doi:10.1088/0034-4885/68/8/R01
def computeXSPH(fluidPositions, fluidVelocities, fluidDensity, fluidAreas, particleSupport, xsphConstant, fluidNeighbors, fluidRadialDistances):
    i = fluidNeighbors[0,:]
    j = fluidNeighbors[1,:]

    pairwiseXSPH = xsphConstant * fluidAreas[j] / ( fluidDensity[i] + fluidDensity[j]) * 2 \
                * (fluidVelocities[j] - fluidVelocities[i]) \
                * kernel(fluidRadialDistances, particleSupport) 
    
    xsphUpdate = scatter_sum(pairwiseXSPH, fluidNeighbors[0], dim=0, dim_size = fluidPositions.shape[0])
    
    return xsphUpdate

# Function to sample particles such that their density equals a desired PDF
def samplePDF(pdf, n = 2048, numParticles = 1024, plot = False, randomSampling = False):
    x = np.linspace(-1,1,n)
    if plot:
        fig, axis = plt.subplots(1, 1, figsize=(9,6), sharex = False, sharey = False, squeeze = False)

    n = 2048
    xs = np.linspace(-1,1,n)

    if plot:
        axis[0,0].plot(xs, pdf(xs))

    normalized_pdf = lambda x: pdf(x) / np.sum(pdf(np.linspace(-1,1,n)))
    if plot:
        axis[0,0].plot(xs, normalized_pdf(xs))
        axis[0,0].axhline(0,ls= '--', color = 'black')


    xs = np.linspace(-1,1,n)
    fxs = normalized_pdf(xs)
    sampled_cdf = np.cumsum(fxs) - fxs[0]
    sampled_cdf = sampled_cdf / sampled_cdf[-1] 
    inv_cdf = lambda x : np.interp(x, sampled_cdf, np.linspace(-1,1,n))

    samples = np.random.uniform(size = numParticles)
    if not randomSampling:
        samples = np.linspace(0,1,numParticles, endpoint=False)
    sampled = inv_cdf(samples)

    return sampled


# SPH simulation step, returns dudt, dxdt as well as current density and pressure
def computeUpdate(fluidPositions, fluidVelocities, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphCoefficient, particleSupport, dt):
    #  1. Create ghost particles for our boundary conditions
    ghostPositions = createGhostParticles(fluidPositions, minDomain, maxDomain)
    #  2. Find neighborhoods of all particles:
    fluidNeighbors, fluidRadialDistances, fluidDistances = findNeighborhoods(fluidPositions, ghostPositions, particleSupport)
    #  3. Compute \rho using an SPH interpolation
    fluidDensity = computeDensity(fluidPositions, fluidAreas, particleSupport, fluidRadialDistances, fluidNeighbors)
    #  4. Compute the pressure of each particle using an ideal gas EOS
    fluidPressure = (fluidDensity - 1.0) * kappa * restDensity
    #  5. Compute the XSPH term and apply it to the particle velocities:    
    xsphUpdate = computeXSPH(fluidPositions, fluidVelocities, fluidDensity, fluidAreas, particleSupport, xsphCoefficient, fluidNeighbors, fluidRadialDistances)
    #  6. Compute pressure forces and resulting acceleration
    fluidPressureForces = computePressureForces(fluidPositions, fluidDensity, fluidPressure, fluidAreas, particleSupport, restDensity, fluidNeighbors, fluidRadialDistances, fluidDistances)
    fluidAccel = fluidPressureForces # / (fluidAreas * restDensity)
    # 7. Compute kinematic viscosity
    laminarViscosity = computeDiffusion(fluidPositions, fluidVelocities, fluidAreas, fluidDensity, particleSupport, restDensity, diffusionAlpha, diffusionBeta, c0, fluidNeighbors, fluidRadialDistances, fluidDistances) # currently broken for some reason
#     fluidAccel += laminarViscosity
    fluidAccel += xsphUpdate / dt + laminarViscosity
    return fluidAccel, fluidVelocities, fluidDensity, fluidPressure

