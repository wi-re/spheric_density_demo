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

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils.repeat import repeat
import torch
from torch_sparse import SparseTensor
from torch import Tensor, nn
from torch.nn import Parameter

from cutlass import cutlass

import math
import numpy as np
from typing import Any
from typing import List, Tuple, Union

def uniform(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)
     
def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)
def zeros(value: Any):
    constant(value, 0.)
    
basestring = (str, bytes)
def is_list_of_strings(lst):
        if lst and isinstance(lst, list):
            return all(isinstance(elem, basestring) for elem in lst)
        else:
            return False
from cutlass import *
import scipy.optimize

MCache = None

def optimizeWeights2D(weights, basis, periodicity, nmc = 32 * 1024, targetIntegral = 1, windowFn = None, verbose = False):
    global MCache
    M = None
    numWeights = weights.shape[0] * weights.shape[1]    
    
    # print(weights.shape, numWeights)
    normalizedWeights = (weights - torch.sum(weights) / weights.numel())/torch.std(weights)
    if not MCache is None:
        cfg, M = MCache
        w,b,n,p,wfn = cfg
        if not(w == weights.shape and np.all(b == basis) and n == nmc and np.all(p ==periodicity) and wfn == windowFn):
            M = None
    # else:
        # print('no cache')
    if M is None:
        r = torch.sqrt(torch.rand(size=(nmc,1)).to(weights.device).type(torch.float32))
        theta = torch.rand(size=(nmc,1)).to(weights.device).type(torch.float32) *2 * np.pi

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        u = evalBasisFunction(weights.shape[0], x.T, which = basis[0], periodic = periodicity[0])[0,:].mT
        v = evalBasisFunction(weights.shape[1], y.T, which = basis[1], periodic = periodicity[1])[0,:].mT
        
    #     print('u', u.shape, u)
    #     print('v', v.shape, v)
        
        window = weights.new_ones(x.shape[0]) if windowFn is None else windowFn(torch.sqrt(x**2 + y**2))[:,0]
        
        
        nuv = torch.einsum('nu, nv -> nuv', u, v)
        nuv = nuv * window[:,None, None]

    #     print('nuv', nuv.shape, nuv)
        M = np.pi * torch.sum(nuv, dim = 0).flatten().detach().cpu().numpy() / nmc
#     print('M', M.shape, M)
        MCache = ((weights.shape, basis, nmc, periodicity, windowFn), M)

    
    w = normalizedWeights.flatten().detach().cpu().numpy()


    eps = 1e-2
    
    if 'chebyshev' in basis or 'fourier' in basis:        
        res = scipy.optimize.minimize(fun = lambda x: (M.dot(x) - targetIntegral)**2, \
                                      jac = lambda x: 2 * M * (M.dot(x) - targetIntegral), \
                                      hess = lambda x: 2. * np.outer(M,M), x0 = w, \
                                      method ='trust-constr', constraints = None,\
                                      options={'disp': False, 'maxiter':100})
    else:
        sumConstraint = scipy.optimize.NonlinearConstraint(fun = np.sum, lb = -eps, ub = eps)
        stdConstraint = scipy.optimize.NonlinearConstraint(fun = np.std, lb = 1 - eps, ub = 1 + eps)

        res = scipy.optimize.minimize(fun = lambda x: (M.dot(x) - targetIntegral)**2, \
                                      jac = lambda x: 2 * M * (M.dot(x) - targetIntegral), \
                                      hess = lambda x: 2. * np.outer(M,M), x0 = w, \
                                      method ='trust-constr', constraints = [sumConstraint, stdConstraint],\
                                      options={'disp': False, 'maxiter':100})
    result = torch.from_numpy(res.x.reshape(weights.shape)).type(torch.float32).to(weights.device)
    if verbose:
        print('result: ', res)
        print('initial weights:', normalizedWeights)
        print('result weights:',result)
        print('initial:', M.dot(w))
        print('integral:', M.dot(res.x))
        print('sumConstraint:', np.sum(res.x))
        print('stdConstraint:', np.std(res.x))
    return result, res.constr, res.fun, M.dot(w), M.dot(res.x)

def mapToSpherical(positions):
    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(y, x)
    phi = torch.acos(z / (r + 1e-7))
    
    return torch.vstack((r,theta,phi)).mT


def ballToCylinder(positions):
    r = torch.linalg.norm(positions, dim = 1)
    xy = torch.linalg.norm(positions[:,:2], dim = 1)
    absz = torch.abs(positions[:,2])

#     debugPrint(r)
#     debugPrint(xy)
#     debugPrint(absz)

    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]

    termA = torch.zeros_like(positions)

    eps = 1e-7

    xB = x * r / (xy + eps)
    yB = y * r / (xy + eps)
    zB = 3 / 2 * z
    termB = torch.vstack((xB, yB, zB)).mT

    xC = x * torch.sqrt(3 * r / (r + absz + eps))
    yC = y * torch.sqrt(3 * r / (r + absz + eps))
    zC = torch.sign(z) * r
    termC = torch.vstack((xC, yC, zC)).mT

    mapped = torch.zeros_like(positions)

    maskA = r < eps
    maskB = torch.logical_and(torch.logical_not(maskA), 5/4 * z**2 <= x**2 + y**2)
    maskC = torch.logical_and(torch.logical_not(maskA), torch.logical_not(maskB))

    mapped[maskB] = termB[maskB]
    mapped[maskC] = termC[maskC]

#     debugPrint(mapped)
    return mapped
# debugPrint(cylinderPositions)

def cylinderToCube(positions):
    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]
    xy = torch.linalg.norm(positions[:,:2], dim = 1)
    eps = 1e-7

    termA = torch.vstack((torch.zeros_like(x), torch.zeros_like(y), z)).mT
    # debugPrint(termA)

    xB = torch.sign(x) * xy
    yB = 4. / np.pi * torch.sign(x) * xy * torch.atan(y/(x+eps))
    zB = z
    termB = torch.vstack((xB, yB, zB)).mT

    xC = 4. / np.pi * torch.sign(y) * xy * torch.atan(x / (y + eps))
    yC = torch.sign(y) * xy
    zC = z
    termC = torch.vstack((xC, yC, zC)).mT

    maskA = torch.logical_and(torch.abs(x) < eps, torch.abs(y) < eps)
    maskB = torch.logical_and(torch.logical_not(maskA), torch.abs(y) <= torch.abs(x))
    maskC = torch.logical_and(torch.logical_not(maskA), torch.logical_not(maskB))

    # debugPrint(torch.sum(maskA))
    # debugPrint(torch.sum(maskB))
    # debugPrint(torch.sum(maskC))


    mapped = torch.zeros_like(positions)
    mapped[maskA] = termA[maskA]
    mapped[maskB] = termB[maskB]
    mapped[maskC] = termC[maskC]
    
    return mapped

def mapToSpherePreserving(positions):
    cylinderPositions = ballToCylinder(positions)
    cubePositions = cylinderToCube(cylinderPositions)
    return cubePositions

class RbfConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 2,
        size: Union[int, List[int]] = [4, 4],
        coordinateMapping : str = 'cartesian',
        rbf : Union[int, List[int]] = 'linear',
        aggr: str = 'sum',

        linearLayer: bool = False,
        feedThrough: bool = False,
        # biasOffset: bool = False,

        preActivation = None,
        postActivation = None,

        bias = True,

        # initializer = torch.nn.init.xavier_uniform_,

        initializer = torch.nn.init.uniform_,

        
        batch_size = [16,16],
        windowFn = None,
        normalizeWeights = False,
        normalizationFactor = None,
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)      
        # self.aggr = aggr
        # assert self.aggr in ['add', 'mean', 'max', None]

        # self.flow = flow
        # assert self.flow in ['source_to_target', 'target_to_source']

        # self.node_dim = node_dim

        # self.inspector = Inspector(self)
        # self.inspector.inspect(self.message)
        # self.inspector.inspect(self.aggregate, pop_first=True)
        # self.inspector.inspect(self.message_and_aggregate, pop_first=True)
        # self.inspector.inspect(self.update, pop_first=True)

        self.__user_args__ = self.inspector.keys(
            ['message', 'aggregate', 'update']).difference(self.special_args)
        self.__fused_user_args__ = self.inspector.keys(
            ['message_and_aggregate', 'update']).difference(self.special_args)

        # Support for "fused" message passing.
        self.fuse = self.inspector.implements('message_and_aggregate')

        # Support for GNNExplainer.
        self.__explain__ = False
        self.__edge_mask__ = None  
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dim = dim
        self.coordinateMapping = coordinateMapping
        # print('coordinate mapping', self.coordinateMapping)
        self.size = size if isinstance(size, list) else repeat(size, dim)
        self.rbfs = rbf if is_list_of_strings(rbf) else [rbf] * dim
        self.periodic = [False, False] if coordinateMapping != 'polar' else [False,True]
        self.initializer = initializer
        self.batchSize = batch_size
        self.feedThrough = feedThrough
        self.preActivation = None if preActivation is None else getattr(nn.functional, preActivation)
        self.postActivation = None if postActivation is None else getattr(nn.functional, postActivation)
        self.windowFn = windowFn
        self.use_bias = bias
        # print('Creating layer %d -> %d features'%( in_channels, out_channels))
        # print('For dimensionality: %d'% dim)
        # print('Parameters:')
        # print('\tRBF: ', self.rbfs)
        # print('\tSize: ', self.size)
        # print('\tPeriodic: ', self.periodic)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.use_bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.K = torch.tensor(self.size).prod().item()
        if dim == 1:
            self.weight = Parameter(torch.Tensor(self.size[0], in_channels[0], out_channels))
        if dim == 2:
            self.weight = Parameter(torch.Tensor(self.size[0],self.size[1], in_channels[0], out_channels))
        if dim == 3:
            self.weight = Parameter(torch.Tensor(self.size[0],self.size[1], self.size[2], in_channels[0], out_channels))
        initializer(self.weight, -0.05, 0.05)
        with torch.no_grad():
            if self.rbfs[0] in ['chebyshev', 'fourier', 'gabor']:
                for i in range(self.dim):
                    if len(self.rbfs) == 1:
                        self.weight[i] *= np.exp(-i)
                    if len(self.rbfs) == 2:
                        self.weight[i,:] *= np.exp(-i)
                    if len(self.rbfs) == 3:
                        self.weight[i,:,:] *= np.exp(-i)
            if len(self.rbfs) > 1 and self.rbfs[1] in ['chebyshev', 'fourier', 'gabor']:
                for i in range(self.dim):
                    if len(self.rbfs) == 2:
                        self.weight[:,i] *= np.exp(-i)
                    if len(self.rbfs) == 3:
                        self.weight[:,i,:] *= np.exp(-i)
            if len(self.rbfs) > 2 and self.rbfs[2] in ['chebyshev', 'fourier', 'gabor']:
                for i in range(self.dim):
                    self.weight[:,:,i] = self.weight[:,:,i] * np.exp(-i)
            if normalizeWeights:
                if len(self.rbfs) == 2:
                    print('Starting normalization')
                    for i in range(in_channels[0]):
                        for j in range(out_channels):
                            newWeights, _, _, init, final = optimizeWeights2D(weights = self.weight[:,:,i,j].detach(),\
                                                                            basis = self.rbfs, periodicity = self.periodic, \
                                                                            nmc = 32*1024, targetIntegral = 1/in_channels[0], \
                                                                            windowFn = self.windowFn, verbose = False) 
                            self.weight[:,:,i,j] = newWeights
                            print('Normalizing [%2d x %2d]: %1.4e => %1.4e (target: %1.4e)' %(i,j, init, final, 1/in_channels[0]))

                            # self.weight[:,:,i,j] /= in_channels[0]
                    print('Done with normalization\n------------------------------------------')

        self.root_weight = linearLayer
        if linearLayer:
            self.lin = Linear(in_channels[1], out_channels, bias=self.use_bias,
                              weight_initializer= 'uniform')

        # if biasOffset:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # if not isinstance(self.weight, nn.UninitializedParameter):
            # size = self.weight.size(0) * self.weight.size(1)
            # self.initializer(self.weight)
        if self.root_weight:
            self.lin.reset_parameters()
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        # print('x', x[0].shape, x)
        # print('edge_index', edge_index.shape, edge_index)
        # print('edge_attr', edge_attr.shape, edge_attr)
        # print('Size', Size)
        # if args.cutlad:
            # out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        # else:
        x_i, x_j = x
        edge_weights = None
        if not(self.windowFn is None):
            edge_weights = self.windowFn(torch.linalg.norm(edge_attr, axis = 1))

        positions = torch.hstack((edge_attr, torch.zeros(edge_attr.shape[0],1, device = edge_attr.device, dtype = edge_attr.dtype)))
        if self.coordinateMapping == 'polar':
            spherical = mapToSpherical(positions)
            mapped = torch.vstack((spherical[:,0] * 2. - 1.,spherical[:,1] / np.pi)).mT
        if self.coordinateMapping == 'cartesian':
            mapped = edge_attr
        if self.coordinateMapping == 'preserving':
            cubePositions = mapToSpherePreserving(positions)
            mapped = torch.vstack((cubePositions[:,0],cubePositions[:,1] / np.pi)).mT
        convolution = cutlass.apply
        out = convolution(edge_index, x_i, x_j, mapped, edge_weights, self.weight, 
                                  x_i.shape[0], self.node_dim,
                              self.size , self.rbfs, self.periodic, 
                              self.batchSize[0],self.batchSize[1]) 

        # out = self.propagate2(edge_index, x=x, edge_attr=edge_attr, size=size)
        

#         print('out: ', out.shape, out)

        x_r = x[1]
        if self.preActivation is not None:
            out = self.preActivation(out)

        if x_r is not None and self.root_weight:
            out = out + self.lin(x_r) if self.preActivation is not None else self.preActivation(self.lin(x_r))
        if self.bias is not None:
            out = out + self.bias
        if self.feedThrough:
            out = out + x_r if self.preActivation is not None else self.preActivation(x_r)
        if self.postActivation is not None:
            out = self.postActivation(out)
        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.dim == 1:
            u = evalBasisFunction(self.size[0], edge_attr[:,0], which=self.rbfs[0], periodic = self.periodic[0]).T
             
            return torch.einsum('nu, uio,ni -> no',u,self.weight, x_j)
        if self.dim == 2:
            u = evalBasisFunction(self.size[0], edge_attr[:,0], which=self.rbfs[0], periodic = self.periodic[0]).T
            v = evalBasisFunction(self.size[1], edge_attr[:,1], which=self.rbfs[1], periodic = self.periodic[1]).T
            
            return torch.einsum('nu, nv, uvio,ni -> no',u,v,self.weight, x_j)
        if self.dim == 3:
            u = evalBasisFunction(self.size[0], edge_attr[:,0], which=self.rbfs[0], periodic = self.periodic[0]).T
            v = evalBasisFunction(self.size[1], edge_attr[:,1], which=self.rbfs[1], periodic = self.periodic[1]).T
            w = evalBasisFunction(self.size[2], edge_attr[:,1], which=self.rbfs[1], periodic = self.periodic[2]).T
            
            return torch.einsum('nu, nv, uvio,ni -> no',u,v,w,self.weight, x_j)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, dim={self.dim})')
    
    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if isinstance(edge_index, Tensor):
            assert edge_index.dtype == torch.long
            assert edge_index.dim() == 2
            assert edge_index.size(0) == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        elif isinstance(edge_index, SparseTensor):
            if self.flow == 'target_to_source':
                raise ValueError(
                    ('Flow direction "target_to_source" is invalid for '
                     'message propagation via `torch_sparse.SparseTensor`. If '
                     'you really want to make use of a reverse message '
                     'passing flow, pass in the transposed sparse tensor to '
                     'the message passing module, e.g., `adj_t.t()`.'))
            the_size[0] = edge_index.sparse_size(1)
            the_size[1] = edge_index.sparse_size(0)
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `torch.LongTensor` of '
             'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
             'argument `edge_index`.'))
    
    def __collect__(self, args, edge_index, size, kwargs):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                dim = 0 if arg[-2:] == '_j' else 1
                data = kwargs.get(arg[:-2], Parameter.empty)

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self.__set_size__(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self.__set_size__(size, dim, data)
                    data = self.__lift__(data, edge_index,
                                         j if arg[-2:] == '_j' else i)

                out[arg] = data

        if isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]
            out['ptr'] = None
        elif isinstance(edge_index, SparseTensor):
            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = edge_index.storage.row()
            out['edge_index_j'] = edge_index.storage.col()
            out['ptr'] = edge_index.storage.rowptr()
            out['edge_weight'] = edge_index.storage.value()
            out['edge_attr'] = edge_index.storage.value()
            out['edge_type'] = edge_index.storage.value()

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[1] or size[0]
        out['size_j'] = size[0] or size[1]
        out['dim_size'] = out['size_i']

        return out

    def propagate2(self, edge_index: Adj, size: Size = None, **kwargs):
        decomposed_layers = 1 if self.explain else self.decomposed_layers

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self.__check_input__(edge_index, size)

        if decomposed_layers > 1:
            user_args = self.__user_args__
            decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
            decomp_kwargs = {
                a: kwargs[a].chunk(decomposed_layers, -1)
                for a in decomp_args
            }
            decomp_out = []

        for i in range(decomposed_layers):
            # if decomposed_layers > 1:
                # for arg in decomp_args:
                    # kwargs[arg] = decomp_kwargs[arg][i]

            # coll_dict = self.__collect__(self.__user_args__, edge_index,
                                            # size, kwargs)

            # msg_kwargs = self.inspector.distribute('message', coll_dict)
            # for hook in self._message_forward_pre_hooks.values():
                # res = hook(self, (msg_kwargs, ))
                # if res is not None:
                    # msg_kwargs = res[0] if isinstance(res, tuple) else res
                    # 
            # aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            
            convolution = cutlass.apply
            
            inFeatures = kwargs['x'][0]
            
            edge_weights = None
            if not(self.windowFn is None):
                edge_weights = self.windowFn(torch.linalg.norm(kwargs['edge_attr'], axis = 1))
                # print(torch.linalg.norm(kwargs['edge_attr'], axis = 1))
                # print(edge_weights.shape)
                # print(edge_weights)
                # print(inFeatures.shape)
                # inFeatures = inFeatures * window[:,None]
                # print(inFeatures.shape)


            positions = torch.hstack((kwargs['edge_attr'], torch.zeros(kwargs['edge_attr'].shape[0],1, device = kwargs['edge_attr'].device, dtype = kwargs['edge_attr'].dtype)))
            if self.coordinateMapping == 'polar':
                spherical = mapToSpherical(positions)
                mapped = torch.vstack((spherical[:,0] * 2. - 1.,spherical[:,1] / np.pi)).mT
            if self.coordinateMapping == 'cartesian':
                mapped = kwargs['edge_attr']
            if self.coordinateMapping == 'preserving':
                cubePositions = mapToSpherePreserving(positions)
                mapped = torch.vstack((cubePositions[:,0],cubePositions[:,1] / np.pi)).mT


            out = convolution(edge_index, kwargs['x'][0], kwargs['x'][1], mapped, edge_weights, self.weight, 
                                            size[0], self.node_dim,
                                        self.size , self.rbfs, self.periodic, 
                                        self.batchSize[0],self.batchSize[1])

        #     for hook in self._aggregate_forward_hooks.values():
        #         res = hook(self, (aggr_kwargs, ), out)
        #         if res is not None:
        #             out = res

        #     update_kwargs = self.inspector.distribute('update', coll_dict)
        #     out = self.update(out, **update_kwargs)

        #     if decomposed_layers > 1:
        #         decomp_out.append(out)

        #     if decomposed_layers > 1:
        #         out = torch.cat(decomp_out, dim=-1)

        # for hook in self._propagate_forward_hooks.values():
        #     res = hook(self, (edge_index, size, kwargs), out)
        #     if res is not None:
        #         out = res

        return out