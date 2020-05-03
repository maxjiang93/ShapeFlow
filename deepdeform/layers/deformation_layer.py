import torch
from torch import nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_regular

import numpy as np


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
    "leakyrelu": nn.LeakyReLU(),
    "tanh10x": Lambda(lambda x: torch.tanh(10*x)),
}


class ImNet(nn.Module):
    """ImNet layer pytorch implementation.
    """

    def __init__(self, dim=3, in_features=32, out_features=4, nf=32,
                 nonlinearity='leakyrelu'):
        """Initialization.
        Args:
          dim: int, dimension of input points.
          in_features: int, length of input features (i.e., latent code).
          out_features: number of output features.
          nf: int, width of the second to last layer.
          activation: tf activation op.
          name: str, name of the layer.
        """
        super(ImNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features
        self.nf = nf
        self.activ = NONLINEARITIES[nonlinearity]
        self.fc0 = nn.Linear(self.dimz, nf*16)
        self.fc1 = nn.Linear(nf*16 + self.dimz, nf*8)
        self.fc2 = nn.Linear(nf*8 + self.dimz, nf*4)
        self.fc3 = nn.Linear(nf*4 + self.dimz, nf*2)
        self.fc4 = nn.Linear(nf*2 + self.dimz, nf*1)
        self.fc5 = nn.Linear(nf*1, out_features)
        self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
        self.fc = nn.ModuleList(self.fc)

    def forward(self, x):
        """Forward method.
        Args:
          x: `[batch_size, dim+in_features]` tensor, inputs to decode.
        Returns:
          output through this layer of shape [batch_size, out_features].
        """
        x_tmp = x
        for dense in self.fc[:4]:
            x_tmp = self.activ(dense(x_tmp))
            x_tmp = torch.cat([x_tmp, x], dim=-1)
        x_tmp = self.activ(self.fc4(x_tmp))
        x_tmp = self.fc5(x_tmp)
        return x_tmp
    
    
class VanillaNet(nn.Module):
    """Vanilla mpl pytorch implementation.
    """

    def __init__(self, dim=3, in_features=32, out_features=3, nf=50, nlayers=4,
                 nonlinearity='leakyrelu'):
        """Initialization.
        Args:
          dim: int, dimension of input points.
          in_features: int, length of input features (i.e., latent code).
          out_features: number of output features.
          nf: int, width of the second to last layer.
          nlayers: int, number of layers in mlp (inc. input/output layers).
          activation: tf activation op.
          name: str, name of the layer.
        """
        super(VanillaNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features
        self.nf = nf
        self.nlayers = nlayers
        self.activ = NONLINEARITIES[nonlinearity]
        modules = [nn.Linear(dim + in_features, nf), self.activ]
        assert(nlayers >= 2)
        for i in range(nlayers-2):
            modules += [nn.Linear(nf, nf), self.activ]
        modules += [nn.Linear(nf, out_features)]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        """Forward method.
        Args:
          x: `[batch_size, dim+in_features]` tensor, inputs to decode.
        Returns:
          output through this layer of shape [batch_size, out_features].
        """
        return self.net(x)


class DeformationFlowNetwork(nn.Module):
    def __init__(self, dim=3, latent_size=1, nlayers=4, width=50, nonlinearity='leakyrelu', arch='imnet'):
        """Intialize deformation flow network.
        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          nlayers: int, number of neural network layers. >= 2.
          width: int, number of neurons per hidden layer. >= 1.
        """
        super(DeformationFlowNetwork, self).__init__()
        self.dim = dim
        self.latent_size = latent_size
        self.nlayers = nlayers
        self.width = width
        self.nonlinearity = nonlinearity
        
        self.arch = arch
        assert(arch in ['imnet', 'vanilla'])
        if arch == 'imnet':
            self.net = ImNet(dim=dim, in_features=latent_size, out_features=dim, 
                             nf=width, nonlinearity=nonlinearity)
        else:  # vanilla
            self.net = VanillaNet(dim=dim, in_features=latent_size, out_features=dim,
                                 nf=width, nlayers=nlayers, nonlinearity=nonlinearity)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, latent_vector, points):
        """
        Args:
          latent_vector: tensor of shape [batch, latent_size], latent code for each shape
          points: tensor of shape [batch, num_points, dim], points representing each shape
        Returns:
          velocities: tensor of shape [batch, num_points, dim], velocity at each point
        """
        latent_vector = latent_vector.unsqueeze(1).expand(-1, points.shape[1], -1)  # [batch, num_points, latent_size]
        points_latents = torch.cat((points, latent_vector), axis=-1)  # [batch, num_points, dim + latent_size]
        b, n, d = points_latents.shape
        res = self.net(points_latents.reshape([-1, d]))
        res = res.reshape([b, n, self.dim])
        return res
    
    
class ConformalDeformationFlowNetwork(nn.Module):
    def __init__(self, dim=3, latent_size=1, nlayers=4, width=50, nonlinearity='softplus',
                 output_scalar=False, arch='imnet'):
        """Intialize conformal deformation flow network w/ irrotational flow.
        
        The network produces a scalar field Phi(x,y,z,t), and the velocity
        field is represented as the gradient of Phi.
        v = \nabla\Phi
        The gradients can be efficiently computed as the Jacobian through backprop.
        
        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          nlayers: int, number of neural network layers. >= 2.
          width: int, number of neurons per hidden layer. >= 1.
        """
        super(ConformalDeformationFlowNetwork, self).__init__()
        self.dim = dim
        self.latent_size = latent_size
        self.nlayers = nlayers
        self.width = width
        self.nonlinearity = nonlinearity
        self.output_scalar = output_scalar
        
        self.scale = nn.Parameter(torch.ones(1) * 1e-1)
        
        nlin = NONLINEARITIES[nonlinearity]
            
        modules = [nn.Linear(dim + latent_size, width), nlin]
        for i in range(nlayers-2):
            modules += [nn.Linear(width, width), nlin]
        modules += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*modules)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, latent_vector, points):
        """
        Args:
          latent_vector: tensor of shape [batch, latent_size], latent code for each shape
          points: tensor of shape [batch, num_points, dim], points representing each shape
        Returns:
          velocities: tensor of shape [batch, num_points, dim], velocity at each point
        """
        latent_vector = latent_vector.unsqueeze(1).expand(-1, points.shape[1], -1)  # [batch, num_points, latent_size]
        b, n, l = latent_vector.shape
        d = l + self.dim
        latent_flat = latent_vector.reshape([-1, l])
        points_flat = points.reshape([-1, self.dim])
        points_flat_ = torch.autograd.Variable(points_flat)
        points_flat_.requires_grad = True
        points_latents = torch.cat((points_flat_, latent_flat), axis=-1)  # [batch*num_points, dim + latent_size]
        phi = self.net(points_latents) * self.scale
        vel_flat = torch.autograd.grad(phi, points_flat_, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        vel = vel_flat.reshape(points.shape)
        if self.output_scalar:
            return vel, phi
        else:
            return vel

    
class DeformationSignNetwork(nn.Module):
    def __init__(self, latent_size=1, nlayers=3, width=20, nonlinearity='tanh'):
        """Initialize deformation sign network.
        Args:
          latent_size: int, size of latent space. >= 1.
          nlayers: int, number of neural network layers. >= 2.
          width: int, number of neurons per hidden layer. >= 1.
        """
        super(DeformationSignNetwork, self).__init__()
        self.latent_size = latent_size
        self.nlayers = nlayers
        self.width = width
        
        nlin = NONLINEARITIES[nonlinearity]
        modules = [nn.Linear(latent_size, width, bias=False), nlin]
        for i in range(nlayers-2):
            modules += [nn.Linear(width, width, bias=False), nlin]
        modules += [nn.Linear(width, 1, bias=False), nlin]
        self.net = nn.Sequential(*modules)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)

    def forward(self, dir_vector):
        """
        Args:
          dir_vector: tensor of shape [batch, latent_size], latent direction.
        Returns:
          signs: tensor of shape [batch, 1, 1]
        """
        dir_vector = dir_vector / (torch.norm(dir_vector, dim=-1, keepdim=True) + 1e-6)  # normalize
        signs = self.net(dir_vector).unsqueeze(-1)
        return signs

    
class NeuralFlowModel(nn.Module):
    def __init__(self, dim=3, latent_size=1, f_nlayers=4, f_width=50, 
                 s_nlayers=3, s_width=20, nonlinearity='relu', conformal=False, arch='imnet', no_sign_net=False):
        super(NeuralFlowModel, self).__init__()
        if conformal:
            model = ConformalDeformationFlowNetwork
            
        else:
            model = DeformationFlowNetwork
        self.no_sign_net = no_sign_net
        self.flow_net = model(dim=dim, latent_size=latent_size, 
                              nlayers=f_nlayers, width=f_width,
                              nonlinearity=nonlinearity, arch=arch)
        if not no_sign_net:
            self.sign_net = DeformationSignNetwork(latent_size=latent_size, 
                                                   nlayers=s_nlayers, width=s_width)
        self.latent_source = None
        self.latent_target = None
        self.latent_updated = False
        self.conformal = conformal
        self.arch = arch
        self.encoder = None
        self.lat_params = None
        self.scale = nn.Parameter(torch.ones(1) * 1e-3)
        
    def add_encoder(self, encoder):
        self.encoder = encoder
        
    def add_lat_params(self, lat_params):
        self.lat_params = lat_params
        
    def get_lat_params(self, idx):
        assert(self.lat_params is not None)
        return self.lat_params[idx]
    
    def update_latents(self, latent_sequence):
        """
        Args:
            latent_sequence: long or float tensor of shape [batch, nsteps, latent_size].
                             sequence of latents along deformation path.
                             if long, index into self.lat_params to retrieve latents.
        """
        bs, ns, d = latent_sequence.shape
        dev = latent_sequence.device
        self.latent_sequence = latent_sequence
        self.latent_seq_len = torch.norm(self.latent_sequence[:, 1:] - self.latent_sequence[:, :-1],
                                         dim=-1)  # [batch, nsteps-1]
        self.latent_seq_len_sum = torch.sum(self.latent_seq_len, dim=1)  # [batch]
        self.latent_seq_weight = self.latent_seq_len / self.latent_seq_len_sum[:, None]  # [batch, nsteps-1]
        self.latent_seq_bins = torch.cumsum(self.latent_seq_weight, dim=1)  # [batch, nsteps-1]
        self.latent_seq_bins = torch.cat([torch.zeros([bs, 1], device=dev), 
                                          self.latent_seq_bins], dim=1)  # [batch, nsteps]
        self.latent_updated = True
        
    def latent_at_t(self, t, return_sign=False):
        t = t.to(self.latent_seq_bins.device)
        # find out which bin this t falls into
        bin_mask = (t > self.latent_seq_bins[:, :-1]) * (t < self.latent_seq_bins[:, 1:])
        # logical and
        
        bin_mask = bin_mask.float()  
        bin_idx = torch.argmax(bin_mask, dim=1)  # [batch,]
        batch_idx = torch.arange(bin_idx.shape[0]).to(bin_idx.device)
        
        # find the interpolation coefficient between the latents at the two ends of the bin
        t0 = self.latent_seq_bins[batch_idx, bin_idx]
        t1 = self.latent_seq_bins[batch_idx, bin_idx+1]  # [batch]
        alpha = (t - t0) / (t1 - t0)  # [batch]
        latent_t0 = self.latent_sequence[batch_idx, bin_idx]  # [batch, latent_size]
        latent_t1 = self.latent_sequence[batch_idx, bin_idx+1]  # [batch, latent_size]
        latent_val = latent_t0 + alpha[:, None] * (latent_t1 - latent_t0)
        latent_dir = (latent_t1 - latent_t0) / torch.norm(latent_t1 - latent_t0, 
                                                          dim=1, keepdim=True)
        zeros = torch.zeros_like(latent_t0)
        outward = torch.norm(latent_t0 - zeros, dim=1) < 1e-6  # [batch]
        sign = (outward.float() - 0.5) * 2
        
        return latent_val, latent_dir, sign
    
    def forward(self, t, points):
        """
        Args:
          t: float, deformation parameter between 0 and 1.
          points: [batch, num_points, dim]
        Returns:
          vel: [batch, num_points, dim]
        """
        # reparametrize eval along latent path as a function of a single scalar t
        if not self.latent_updated:
            raise RuntimeError('Latent not updated. '
                               'Use .update_latents() to update the source and target latents.')
        
        latent_val, latent_dir, sign = self.latent_at_t(t)
        sign = sign[:, None, None] * self.scale
        flow = self.flow_net(latent_val, points)  # [batch, num_pints, dim]
        # normalize velocity based on time space proportional to latent difference
        flow *= self.latent_seq_len_sum[:, None, None]
        if not self.no_sign_net:
            sign = self.sign_net(latent_dir)
        return flow * sign
    
    
class NeuralFlowDeformer(nn.Module):
    def __init__(self, dim=3, latent_size=1, f_nlayers=4, f_width=50, 
                 s_nlayers=3, s_width=20, method='dopri5', nonlinearity='leakyrelu', 
                 arch='imnet', conformal=False, adjoint=True, atol=1e-5, rtol=1e-5,
                 via_hub=False, no_sign_net=False):
        """Initialize. The parameters are the parameters for the Deformation Flow network.
        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          f_nlayers: int, number of neural network layers for flow network. >= 2.
          f_width: int, number of neurons per hidden layer for flow network. >= 1.
          s_nlayers: int, number of neural network layers for sign network. >= 2.
          s_width: int, number of neurons per hidden layer for sign network. >= 1.
          arch: str, architecture, choice of 'imnet' / 'vanilla'
          adjoint: bool, whether to use adjoint solver to backprop gadient thru odeint.
          rtol, atol: float, relative / absolute error tolerence in ode solver.
          via_hub: will perform transformation via hub-and-spokes configuration.
                   only useful if latent_sequence is torch.long
        """
        super(NeuralFlowDeformer, self).__init__()
        self.method = method
        self.conformal = conformal
        self.arch = arch
        self.adjoint = adjoint
        self.odeint = odeint_adjoint if adjoint else odeint_regular
        self.timing = torch.from_numpy(np.array([0, 1]).astype('float32'))
        self.rtol = rtol
        self.atol = atol
        self.via_hub = via_hub

        self.net = NeuralFlowModel(dim=dim, latent_size=latent_size, 
                                   f_nlayers=f_nlayers, f_width=f_width,
                                   s_nlayers=s_nlayers, s_width=s_width,
                                   arch=arch, conformal=conformal, 
                                   nonlinearity=nonlinearity, no_sign_net=no_sign_net)
        
    @property
    def adjoint(self):
        return self.__adjoint
    
    
    @adjoint.setter
    def adjoint(self, isadjoint):
        assert(isinstance(isadjoint, bool))
        self.__adjoint = isadjoint
        self.odeint = odeint_adjoint if isadjoint else odeint_regular
        
    def add_encoder(self, encoder):
        self.net.add_encoder(encoder)
        
    def add_lat_params(self, lat_params):
        self.net.add_lat_params(lat_params)
        
    def get_lat_params(self, idx):
        return self.net.get_lat_params(idx)
  
    def forward(self, points, latent_sequence):
        """Forward transformation (source -> latent_path -> target).
        
        To perform backward transformation, simply switch the order of the lat codes.
        
        Args:
          points: [batch, num_points, dim]
          latent_sequence: float tensor of shape [batch, nsteps, latent_size], 
                           ------- or -------
                           long tensor of shape [batch, nsteps]
                           sequence of latents along deformation path.
                           if long, index into self.lat_params to retrieve latents.
        Returns:
          points_transformed: [batch, num_points, dim]
        """
        if latent_sequence.dtype == torch.long:
            latent_sequence = self.get_lat_params(latent_sequence)  # [nsteps, batch, lat_dim]
            if self.via_hub:
                assert(latent_sequence.shape[1] == 2)
                zeros = torch.zeros_like(latent_sequence[:, :1])  
                lat0 = latent_sequence[:, 0:1]
                lat1 = latent_sequence[:, 1:2]
                latent_sequence = torch.cat([lat0, zeros, lat1], dim=1)  # [batch, nsteps=3, lat_dim]
        self.net.update_latents(latent_sequence)
        points_transformed = self.odeint(self.net, points, self.timing, 
                                         method=self.method, rtol=self.rtol, atol=self.atol)[1]
        return points_transformed

