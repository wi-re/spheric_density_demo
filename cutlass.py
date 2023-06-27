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
#

import torch
from torch.profiler import record_function
import numpy as np
from typing import Dict, Optional

# ------ Beginning of scatter functionality ------ #
# Scatter summation functionality based on pytorch geometric scatter functionality
# This is included here to make the code independent of pytorch geometric for portability
# Note that pytorch geometric is licensed under an MIT licenses for the PyG Team <team@pyg.org>
@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
# ------ End of scatter functionality ------ #

# Spacing for basis functions
def getSpacing(n, periodic = False):
    if n == 1:
        return 2.
    else:
        return 2. / n if periodic else 2./(n-1)
    
# Function that returns the distance between a given set of points and a set of basis function centers
# Caches the basis function center positions for computational efficiency
centroidCache = {False:{'cuda':{},'cpu':{}},True:{'cuda':{},'cpu':{}}}
def getDistancesRel(n, x, periodic = False):
    if n in centroidCache[periodic][x.device.type]:
        centroids = centroidCache[periodic][x.device.type][n]
        if periodic:
            spacing = getSpacing(n, True)
            offset = -1 + spacing / 2.
            ra = torch.unsqueeze(x,axis=0) - centroids
            rb = torch.unsqueeze(x,axis=0) - centroids - 2.
            rc = torch.unsqueeze(x,axis=0) - centroids + 2.
            return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        else:
            spacing = getSpacing(n, False)
            r = torch.unsqueeze(x,axis=0) - centroids
            return r  / spacing


    if periodic:
        spacing = getSpacing(n, True)
        centroids = torch.unsqueeze(torch.linspace(-1.,1.,n+1, device = x.device)[:n],axis=1)
        centroidCache[periodic][x.device.type][n] = centroids

        ra = torch.unsqueeze(x,axis=0) - centroids
        rb = torch.unsqueeze(x,axis=0) - centroids - 2.
        rc = torch.unsqueeze(x,axis=0) - centroids + 2.
        return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        
    spacing = getSpacing(n, False)
    
    centroids = torch.linspace(-1.,1.,n, device = x.device) if n > 1 else torch.tensor([0.], device = x.device)
    centroids = torch.unsqueeze(centroids, axis = 1)
    centroidCache[periodic][x.device.type][n] = centroids
    r = torch.unsqueeze(x,axis=0) - centroids
    return r  / spacing

# Evaluate a set of radial basis functions with a variety of options
def evalRBFSeries(n, x, which = 'linear', epsilon = 1., periodic = False, adjustSpacing = False, normalized = False):   
    k = int(epsilon)
    if adjustSpacing:
        if which == 'gaussian' or which == 'inverse_quadric' or which == 'inverse_multiquadric' or 'spline' in which  or 'wendland' in which:
            x = x * (1 - 2/n)
        if which == 'bump':
            x = x * (1 - 4/n)
    
    rRel = getDistancesRel(n, x, periodic)
    r = torch.abs(rRel)
    if n == 1:
        return torch.ones_like(r)
    
    cpow = lambda x, p: torch.maximum(x, torch.zeros_like(r))**p
    
    funLib = {
        'linear': lambda r:  torch.clamp(1. - r / epsilon,0,1),
        'gaussian': lambda r:  torch.exp(-(epsilon * r)**2),
        'multiquadric': lambda r: torch.sqrt(1. + (epsilon * r) **2),
        'inverse_quadric': lambda r: 1. / ( 1 + (epsilon * r) **2),
        'inverse_multiquadric': lambda r: 1. / torch.sqrt(1. + (epsilon * r) **2),
        'polyharmonic': lambda r: torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r)),
        'bump': lambda r: torch.where(r < 1./epsilon, torch.exp(-1./(1- (epsilon * r)**2)), torch.zeros_like(r)),
        'cubic_spline': lambda r: cpow(1-r/(epsilon * 1.),3) - 4. * cpow(1/2-r/(epsilon * 1.),3),
        'quartic_spline': lambda r: cpow(1-r/(epsilon * 1.),4) - 5 * cpow(3/5-r/(epsilon * 1.),4) + 10 * cpow(1/5-r/(epsilon * 1.),4),
        'quintic_spline': lambda r: cpow(1-r/(epsilon * 1.),5) - 6 * cpow(2/3-r/(epsilon * 1.),5) + 15 * cpow(1/3-r/(epsilon * 1.),5),
        'wendland2': lambda r: cpow(1 - r/(epsilon * 1.), 4) * (1 + 4 * r/(epsilon * 1.)),
        'wendland4': lambda r: cpow(1 - r/(epsilon * 1.), 6) * (1 + 6 * r/(epsilon * 1.) + 35/3 * (r/(epsilon * 1.))**2),
        'wendland6': lambda r: cpow(1 - r/(epsilon * 1.), 8) * (1 + 8 * r/(epsilon * 1.) + 25 * (r/(epsilon * 1.)) **2 + 32 * (r * (epsilon * 1.))**3),
        'poly6': lambda r: cpow(1 - (r/epsilon)**2, 3),
        'spiky': lambda r: cpow(1 - r/epsilon, 3),
        'square': lambda r: torch.where(torch.logical_and(rRel > -0.5 * epsilon, rRel <= 0.5 * epsilon), torch.ones_like(r), torch.zeros_like(r))
    }
    normalizedFunLib = {
        'linear': lambda r:  torch.clamp(1. - r / epsilon,0,1),
        'gaussian': lambda r:  torch.exp(-(epsilon * r)**2),
        'multiquadric': lambda r: torch.sqrt(1. + (epsilon * r) **2),
        'inverse_quadric': lambda r: 1. / ( 1 + (epsilon * r) **2),
        'inverse_multiquadric': lambda r: 1. / torch.sqrt(1. + (epsilon * r) **2),
        'polyharmonic': lambda r: torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r)),
        'bump': lambda r: torch.where(r < 1./epsilon, torch.exp(-1./(1- (epsilon * r)**2)), torch.zeros_like(r)),
        'cubic_spline': lambda r: cpow(1-r/(epsilon * 1.732051),3) - 4. * cpow(1/2-r/(epsilon * 1.732051),3),
        'quartic_spline': lambda r: cpow(1-r/(epsilon * 1.936492),4) - 5 * cpow(3/5-r/(epsilon * 1.936492),4) + 10 * cpow(1/5-r/(epsilon * 1.732051),4),
        'quintic_spline': lambda r: cpow(1-r/(epsilon * 2.121321),5) - 6 * cpow(2/3-r/(epsilon * 2.121321),5) + 15 * cpow(1/3-r/(epsilon * 2.121321),5),
        'wendland2': lambda r: cpow(1 - r/(epsilon * 1.620185), 4) * (1 + 4 * r/(epsilon * 1.620185)),
        'wendland4': lambda r: cpow(1 - r/(epsilon * 1.936492), 6) * (1 + 6 * r/(epsilon * 1.936492) + 35/3 * (r/(epsilon * 1.936492))**2),
        'wendland6': lambda r: cpow(1 - r/(epsilon * 2.207940), 8) * (1 + 8 * r/(epsilon * 2.207940) + 25 * (r/(epsilon * 2.207940)) **2 + 32 * (r * (epsilon * 2.207940))**3),
        'poly6': lambda r: cpow(1 - (r/epsilon)**2, 3),
        'spiky': lambda r: cpow(1 - r/epsilon, 3),
        'square': lambda r: torch.where(torch.logical_and(rRel > -0.5 * epsilon, rRel <= 0.5 * epsilon), torch.ones_like(r), torch.zeros_like(r))
    }    
    adjustedFunLib = {
        'linear': lambda r:  torch.clamp(1. - r / 1,0,1),
        'gaussian': lambda r:  torch.exp(-(0.9919394235466537 * r)**2),
        'multiquadric': lambda r: torch.sqrt(1. + (1 * r) **2),
        'inverse_quadric': lambda r: 1. / ( 1 + (1.1480214948705423 * r) **2),
        'inverse_multiquadric': lambda r: 1. / torch.sqrt(1. + (1.6382510991695163 * r) **2),
        'polyharmonic': lambda r: torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r)),
        'bump': lambda r: torch.where(r < 1./0.38739618954567656, torch.exp(-1./(1- (0.38739618954567656 * r)**2)), torch.zeros_like(r)),
        'cubic_spline': lambda r: cpow(1-r/(epsilon * 2.009770395701026),3) - 4. * cpow(1/2-r/(epsilon * 2.009770395701026),3),
        'quartic_spline': lambda r: cpow(1-r/(epsilon * 2.4318514899853443),4) - 5 * cpow(3/5-r/(epsilon * 2.4318514899853443),4) + 10 * cpow(1/5-r/(epsilon * 2.4318514899853443),4),
        'quintic_spline': lambda r: cpow(1-r/(epsilon * 2.8903273082559844),5) - 6 * cpow(2/3-r/(epsilon * 2.8903273082559844),5) + 15 * cpow(1/3-r/(epsilon * 2.8903273082559844),5),
        'wendland2': lambda r: cpow(1 - r/(epsilon * 3.6238397655105032), 4) * (1 + 4 * r/(epsilon * 3.6238397655105032)),
        'wendland4': lambda r: cpow(1 - r/(epsilon * 3.7338788470933073), 6) * (1 + 6 * r/(epsilon * 3.7338788470933073) + 35/3 * (r/(epsilon * 3.7338788470933073))**2),
        'wendland6': lambda r: cpow(1 - r/(epsilon * 1.3856863702979971), 8) * (1 + 8 * r/(epsilon * 1.3856863702979971) + 25 * (r/(epsilon * 1.3856863702979971)) **2 + 32 * (r * (epsilon * 1.3856863702979971))**3),
        'poly6': lambda r: cpow(1 - (r/ 2.6936980947728384)**2, 3),
        'spiky': lambda r: cpow(1 - r/3, 3),
        'square': lambda r: torch.where(torch.logical_and(rRel > -0.5 * 1, rRel <= 0.5 * 1), torch.ones_like(r), torch.zeros_like(r))

    }
    
    rbf = funLib[which]
    if adjustSpacing:
        rbf = adjustedFunLib[which]
    if normalized:
        rbf = normalizedFunLib[which]
    res = rbf(r)
    if normalized:
        res = res / torch.sum(res, dim = 0)
    return res
# Evaluate a chebyshev series of the first kind
def evalChebSeries(n,x):
    cs = []
    for i in range(n):
        if i == 0:
            cs.append(torch.ones_like(x))
        elif i == 1:
            cs.append(x)
        else:
            cs.append(2. * x * cs[i-1] - cs[i-2])
    return torch.stack(cs)

# Evaluate a chebyshev series of the second kind
def evalChebSeries2(n,x):
    cs = []
    for i in range(n):
        if i == 0:
            cs.append(torch.ones_like(x))
        elif i == 1:
            cs.append(2 * x)
        else:
            cs.append(2. * x * cs[i-1] - cs[i-2])
    return torch.stack(cs)

# precomputed value for computational efficiency
sqrt_pi_1 = 1. / np.sqrt(np.pi)
# Evaluate a fourier series
def fourier(n, x):
    if n == 0:
        return torch.ones_like(x) / np.sqrt(2. * np.pi)
    elif n % 2 == 0:
        return torch.cos((n // 2 + 1) * x) * sqrt_pi_1
    return torch.sin((n // 2 + 1) * x) * sqrt_pi_1
def evalFourierSeries(n, x):
    fs = []
    for i in range(n):
        fs.append(fourier(i, x))
    return torch.stack(fs)

# Parent function that delegates the call to the corresponding evaluation functions
def evalBasisFunction(n, x, which = 'chebyshev', periodic = False):   
    s = which.split()    
    if s[0] == 'chebyshev':
        return evalChebSeries(n, x)
    if s[0] == 'chebyshev2':
        return evalChebSeries2(n, x)
    if s[0] == 'fourier':
        return evalFourierSeries(n, x * np.pi)
    if s[0] == 'linear':
        return evalRBFSeries(n, x, which = 'linear', epsilon = 1., periodic = periodic)        
    if s[0] == 'rbf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic)     
    if s[0] == 'abf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic, adjustSpacing = True)     
    if s[0] == 'ubf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic, normalized = True)


# This function is used to compute a continuous convolution with an arbitrary radial basis function
# The naming is a reference to NVIDIA's cutlass library which can be used to perform similar tasks for normal
# convolutions and continuous convolutions with linear basii as done in the Open3D codebase used in the original
# continuous convolution paper by Benjamin Ummenhofer
class cutlass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, features_i, features_j, edge_attr, edge_weights, weight, 
                dim_size, dim, size, rbfs, periodic, forwardBatchSize, backwardBatchSize, normalized = False):
        with record_function("cutlass forward step"): 
            ctx.save_for_backward(edge_index, features_i, features_j, edge_attr, edge_weights, weight)
            ctx.dimensions = len(size)
            ctx.dim_size = dim_size
            ctx.dim = dim
            ctx.size = size
            ctx.rbfs = rbfs
            ctx.periodic = periodic
            ctx.forwardBatchSize = forwardBatchSize
            ctx.backwardBatchSize = backwardBatchSize
            ctx.normalized = normalized
            

            with record_function("cutlass forward batchprep"): 
                x_j = features_j[edge_index[1]]
                x_j = x_j if edge_weights is None else x_j * edge_weights[:,None]
                indices = torch.arange(0,edge_attr.shape[0]).to(features_j.device)            
                batches = torch.split(indices, ctx.forwardBatchSize * 1024)
            out = features_i.new_zeros((features_i.shape[0], weight.shape[-1])).type(features_i.dtype)

            for batch in batches:
                if ctx.dimensions == 1:
                    with record_function("cutlass forward batch"): 
                        with record_function("cutlass forward basis"): 
                            u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T

                        if ctx.normalized:
                            normalizationFactor = u.sum(-1)
                            conv = torch.einsum('nu, uio,ni -> no',u,weight, x_j[batch]) * normalizationFactor[:,None]
                        else:
                            conv = torch.einsum('nu, uio,ni -> no',u,weight, x_j[batch])
                        del u
                        out += scatter_sum(conv, index = edge_index[0,batch], dim_size = ctx.dim_size, dim = ctx.dim)
                        del conv
                if ctx.dimensions == 2:
                    with record_function("cutlass forward batch"): 
                        with record_function("cutlass forward basis"): 
                            u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                            v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T

                        with record_function("cutlass forward einsum"): 
                            if ctx.normalized:
                                normalizationFactor = torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                conv = torch.einsum('nu, nv, uvio,ni -> no',u,v,weight, x_j[batch]) * normalizationFactor[:,None]
                            else:
                                conv = torch.einsum('nu, nv, uvio,ni -> no',u,v,weight, x_j[batch])
                        del u,v
                        out += scatter_sum(conv, index = edge_index[0,batch], dim_size = ctx.dim_size, dim = ctx.dim)
                        del conv
            return out
    # needs manual gradients as the auto diff version requires excessive amounts of memory and is computationally slower
    # the mathematical details here will be explained at a later point in a more written out form
    @staticmethod
    def backward(ctx, grad_output):
        with record_function("cutlass backward step"): 
            edge_index, features_i, features_j, edge_attr, edge_weights, weight = ctx.saved_tensors
            
            featureGrad = None
            weightGrad = None
            
            with record_function("cutlass backward batching"): 
                x_j = torch.index_select(features_j, 0, edge_index[1])
                x_j = x_j if edge_weights is None else x_j * edge_weights[:,None]
                gradFeatures = torch.index_select(grad_output, 0, edge_index[0])
                indices = torch.arange(0,edge_attr.shape[0]).to(features_i.device)            
                batches = torch.split(indices, ctx.backwardBatchSize * 1024)
            
            if ctx.needs_input_grad[2] and not ctx.needs_input_grad[5]:  
                with record_function("cutlass backward feature grad"):                        
                    transposedWeights = torch.transpose(weight, -2, -1)     
                    convs = []
                    for batch in batches:
                        if ctx.dimensions == 1:
                            with record_function("cutlass backward feature grad batch"):    
                                with record_function("cutlass backward feature grad basis"):    
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                
                                with record_function("cutlass backward feature grad einsum"):    
                                    if ctx.normalized:
                                        if edge_weights is not None:
                                            normalizationFactor = 1 / u.sum(-1)
                                            convs.append(torch.einsum('nu, n, uio,ni -> no',u, edge_weights[batch], transposedWeights, gradFeatures[batch]) * normalizationFactor[:,None])
                                        else:
                                            normalizationFactor = 1 / torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                            convs.append(torch.einsum('nu, uio,ni -> no',u, transposedWeights, gradFeatures[batch])*normalizationFactor[:,None])
                                    else:       
                                        if edge_weights is not None: 
                                            convs.append(torch.einsum('nu, n, uio,ni -> no',u, edge_weights[batch], transposedWeights, gradFeatures[batch]))
                                        else:
                                            convs.append(torch.einsum('nu, uio,ni -> no',u, transposedWeights, gradFeatures[batch]))
                                del u
                        if ctx.dimensions == 2:
                            with record_function("cutlass backward feature grad batch"):    
                                with record_function("cutlass backward feature grad basis"):    
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                                
                                with record_function("cutlass backward feature grad einsum"):    
                                    if ctx.normalized:
                                        if edge_weights is not None:
                                            normalizationFactor = 1 / torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                            convs.append(torch.einsum('nu, nv, n, uvio,ni -> no',u,v, edge_weights[batch], transposedWeights, gradFeatures[batch]) * normalizationFactor[:,None])
                                        else:
                                            normalizationFactor = 1 / torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                            convs.append(torch.einsum('nu, nv, uvio,ni -> no',u,v, transposedWeights, gradFeatures[batch])*normalizationFactor[:,None])
                                    else:       
                                        if edge_weights is not None: 
                                            convs.append(torch.einsum('nu, nv, n, uvio,ni -> no',u,v, edge_weights[batch], transposedWeights, gradFeatures[batch]))
                                        else:
                                            convs.append(torch.einsum('nu, nv, uvio,ni -> no',u,v, transposedWeights, gradFeatures[batch]))
                                del u,v
                    with record_function("cutlass backward feature grad stacking"):   
                        out = torch.vstack(convs)
                    with record_function("cutlass backward feature grad aggregation"):   
                        featureGrad = scatter_sum(out, index = edge_index[1], dim_size = features_j.shape[0], dim = ctx.dim)       
            if ctx.needs_input_grad[5] and not ctx.needs_input_grad[2]:   
                with record_function("cutlass backward weight grad"):    
                    weightGrad = weight.new_zeros(weight.shape)                    
                    for batch in batches:
                        if ctx.dimensions == 1:
                            with record_function("cutlass backward weight grad batch"):   
                                with record_function("cutlass backward weight grad batch basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T

                                with record_function("cutlass backward weight grad batch einsum"):   
                                    if ctx.normalized:
                                        normalizationFactor = 1 / u.sum(-1)
                                        localGrad = torch.einsum('nu, n, ni, no -> uio', u, normalizationFactor, x_j[batch], gradFeatures[batch])
                                    else:                                        
                                        localGrad = torch.einsum('nu, ni, no -> uio', u, x_j[batch], gradFeatures[batch])
                                    weightGrad += localGrad
                                del u
                        if ctx.dimensions == 2:
                            with record_function("cutlass backward weight grad batch"):   
                                with record_function("cutlass backward weight grad batch basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T

                                with record_function("cutlass backward weight grad batch einsum"):   
                                    if ctx.normalized:
                                        normalizationFactor = 1 / torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                        localGrad = torch.einsum('nu, nv, n, ni, no -> uvio', u, v, normalizationFactor, x_j[batch], gradFeatures[batch])
                                    else:                                        
                                        localGrad = torch.einsum('nu, nv, ni, no -> uvio', u, v,x_j[batch], gradFeatures[batch])
                                    weightGrad += localGrad
                                del u,v
            if ctx.needs_input_grad[2] and ctx.needs_input_grad[5]:  
                with record_function("cutlass backward"):      
                    weightGrad = weight.new_zeros(weight.shape)                    
                    transposedWeights = torch.transpose(weight, -2, -1)        
                    convs = []
                    for batch in batches:
                        if ctx.dimensions == 1:
                            with record_function("cutlass backward batch"):   
                                with record_function("cutlass backward basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                if ctx.normalized:
                                    normalizationFactor = 1 / u.sum(-1)
                                    u = u * normalizationFactor[:,None]
                            with record_function("cutlass backward einsum features"):   

                                if ctx.normalized:
                                    if edge_weights is not None:
                                        normalizationFactor = 1 / u.sum(-1)
                                        convs.append(torch.einsum('nu, n, uio,ni -> no',u, edge_weights[batch], transposedWeights, gradFeatures[batch]) * normalizationFactor[:,None])
                                    else:
                                        normalizationFactor = 1 / u.sum(-1)
                                        convs.append(torch.einsum('nu, uio,ni -> no',u, transposedWeights, gradFeatures[batch])*normalizationFactor[:,None])
                                else:       
                                    if edge_weights is not None: 
                                        convs.append(torch.einsum('nu, n, uio,ni -> no',u, edge_weights[batch], transposedWeights, gradFeatures[batch]))
                                    else:
                                        convs.append(torch.einsum('nu, uio,ni -> no',u, transposedWeights, gradFeatures[batch]))
                            with record_function("cutlass backward einsum grad"):   
                                io = torch.einsum('ni, no -> nio', x_j[batch], gradFeatures[batch])
                                localGrad = torch.einsum('nu, nio -> uio', u, io)
                                weightGrad += localGrad
                        if ctx.dimensions == 2:
                            with record_function("cutlass backward batch"):   
                                with record_function("cutlass backward basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                            with record_function("cutlass backward einsum uvw"):   
                                if ctx.normalized:
                                    normalizationFactor = 1 / torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                    uvw = torch.einsum('nu, nv -> nuv', u, v) * normalizationFactor[:,None,None]
                                else:
                                    uvw = torch.einsum('nu, nv -> nuv', u, v) 
                                del u,v
                            with record_function("cutlass backward einsum features"):   
                                if edge_weights is not None:
                                    convs.append(torch.einsum('nuv, n, uvio,ni -> no',uvw, edge_weights[batch], transposedWeights, gradFeatures[batch]))
                                else:
                                    convs.append(torch.einsum('nuv, uvio,ni -> no',uvw, transposedWeights, gradFeatures[batch]))
                            with record_function("cutlass backward einsum grad"):   
                                io = torch.einsum('ni, no -> nio', x_j[batch], gradFeatures[batch])
                                localGrad = torch.einsum('nuv, nio -> uvio', uvw, io)
                                weightGrad += localGrad
                    with record_function("cutlass backward stacking"):   
                        out = torch.vstack(convs)
                    with record_function("cutlass backward aggregation"):   
                        featureGrad = scatter_sum(out, index = edge_index[1], dim_size = features_j.shape[0], dim = ctx.dim) 
            return None, None, featureGrad, None, None, weightGrad, None, None, None, None, None, None, None, None 
