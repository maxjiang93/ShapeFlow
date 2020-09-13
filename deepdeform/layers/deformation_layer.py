import torch
from torch import nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_regular
from .pde_layer import PDELayer
from .shared_definition import NONLINEARITIES
import numpy as np


class ImNet(nn.Module):
    """ImNet layer pytorch implementation."""

    def __init__(
        self,
        dim=3,
        in_features=32,
        out_features=4,
        nf=32,
        nonlinearity="leakyrelu",
    ):
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
        self.fc0 = nn.Linear(self.dimz, nf * 16)
        self.fc1 = nn.Linear(nf * 16 + self.dimz, nf * 8)
        self.fc2 = nn.Linear(nf * 8 + self.dimz, nf * 4)
        self.fc3 = nn.Linear(nf * 4 + self.dimz, nf * 2)
        self.fc4 = nn.Linear(nf * 2 + self.dimz, nf * 1)
        self.fc5 = nn.Linear(nf * 1, out_features)
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
    """Vanilla mpl pytorch implementation."""

    def __init__(
        self,
        dim=3,
        in_features=32,
        out_features=3,
        nf=50,
        nlayers=4,
        nonlinearity="leakyrelu",
    ):
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
        assert nlayers >= 2
        for i in range(nlayers - 2):
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


def symmetrize(net, latent_vector, points, symm_dim):
    """Make network output symmetric."""
    # query both sides of the symmetric dimension
    points_pos = points
    points_neg = points.clone()
    points_neg[..., symm_dim] = -points_neg[..., symm_dim]
    y_pos = net(latent_vector, points_pos)
    y_neg = net(latent_vector, points_neg)
    y_sym = (y_pos + y_neg) / 2
    y_sym[..., symm_dim] = (y_pos[..., symm_dim] - y_neg[..., symm_dim]) / 2
    return y_sym


class DeformationFlowNetwork(nn.Module):
    def __init__(
        self,
        dim=3,
        latent_size=1,
        nlayers=4,
        width=50,
        nonlinearity="leakyrelu",
        arch="imnet",
        divfree=False,
    ):
        """Intialize deformation flow network.

        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          nlayers: int, number of neural network layers. >= 2.
          width: int, number of neurons per hidden layer. >= 1.
          divfree: bool, paramaterize a divergence free flow.
        """
        super(DeformationFlowNetwork, self).__init__()
        self.dim = dim
        self.latent_size = latent_size
        self.nlayers = nlayers
        self.width = width
        self.nonlinearity = nonlinearity

        self.arch = arch
        self.divfree = divfree

        assert arch in ["imnet", "vanilla"]
        if arch == "imnet":
            self.net = ImNet(
                dim=dim,
                in_features=latent_size,
                out_features=dim,
                nf=width,
                nonlinearity=nonlinearity,
            )
        else:  # vanilla
            self.net = VanillaNet(
                dim=dim,
                in_features=latent_size,
                out_features=dim,
                nf=width,
                nlayers=nlayers,
                nonlinearity=nonlinearity,
            )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                nn.init.constant_(m.bias, val=0)
        if divfree:
            self.curl = self._get_curl_layer()

    def _get_curl_layer(self):
        in_vars = "x, y, z"
        out_vars = "u, v, w"
        eqn_strs = [
            "dif(w, y) - dif(v, z)",
            "dif(u, z) - dif(w, x)",
            "dif(v, x) - dif(u, y)",
        ]
        eqn_names = [
            "curl_x",
            "curl_y",
            "curl_z",
        ]  # a name/identifier for the equations
        curl = PDELayer(
            in_vars=in_vars, out_vars=out_vars
        )  # initialize the pde layer
        for eqn_str, eqn_name in zip(eqn_strs, eqn_names):  # add equations
            curl.add_equation(eqn_str, eqn_name)
        return curl

    def forward(self, latent_vector, points):
        """
        Args:
          latent_vector: tensor of shape [batch, latent_size], latent code for
            each shape
          points: tensor of shape [batch, num_points, dim], points representing
            each shape

        Returns:
          velocities: tensor of shape [batch, num_points, dim], velocity at
            each point
        """
        latent_vector = latent_vector.unsqueeze(1).expand(
            -1, points.shape[1], -1
        )  # [batch, num_points, latent_size]
        # wrapper for pde layer

        def fwd_fn(points):
            """Forward function.

            Where inpt[..., 0], inpt[..., 1], inpt[..., 2] correspond to x, y,
              z and
            out[..., 0], out[..., 1], out[..., 2] correspond to u, v, w
            """
            points_latents = torch.cat(
                (points, latent_vector), axis=-1
            )  # [batch, num_points, dim + latent_size]
            b, n, d = points_latents.shape
            res = self.net(points_latents.reshape([-1, d]))
            res = res.reshape([b, n, self.dim])
            return res

        if self.divfree:
            # return the curl of the velocity field instead
            self.curl.update_forward_method(fwd_fn)
            _, res_dict = self.curl(points)  # res are the equation residues
            res = torch.cat(
                [res_dict["curl_x"], res_dict["curl_y"], res_dict["curl_z"]],
                dim=-1,
            )  # [batch, num_points, dim]
            return res
        else:
            return fwd_fn(points)


class ConformalDeformationFlowNetwork(nn.Module):
    def __init__(
        self,
        dim=3,
        latent_size=1,
        nlayers=4,
        width=50,
        nonlinearity="softplus",
        output_scalar=False,
        arch="imnet",
    ):
        """Intialize conformal deformation flow network w/ irrotational flow.

        The network produces a scalar field Phi(x,y,z,t), and the velocity
        field is represented as the gradient of Phi.
        v = \nabla\Phi  # noqa: W605
        The gradients can be efficiently computed as the Jacobian through
        backprop.

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
        for i in range(nlayers - 2):
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
          latent_vector: tensor of shape [batch, latent_size], latent code for
            each shape
          points: tensor of shape [batch, num_points, dim], points representing
            each shape
        Returns:
          velocities: tensor of shape [batch, num_points, dim], velocity at
            each point
        """
        latent_vector = latent_vector.unsqueeze(1).expand(
            -1, points.shape[1], -1
        )  # [batch, num_points, latent_size]
        b, num_points, lat_size = latent_vector.shape
        latent_flat = latent_vector.reshape([-1, lat_size])
        points_flat = points.reshape([-1, self.dim])
        points_flat_ = torch.autograd.Variable(points_flat)
        points_flat_.requires_grad = True
        points_latents = torch.cat(
            (points_flat_, latent_flat), axis=-1
        )  # [batch*num_points, dim + latent_size]
        phi = self.net(points_latents) * self.scale
        vel_flat = torch.autograd.grad(
            phi,
            points_flat_,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
        )[0]
        vel = vel_flat.reshape(points.shape)
        if self.output_scalar:
            return vel, phi
        else:
            return vel


class DeformationSignNetwork(nn.Module):
    def __init__(
        self, latent_size=1, nlayers=3, width=20, nonlinearity="tanh"
    ):
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
        for i in range(nlayers - 2):
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
        dir_vector = dir_vector / (
            torch.norm(dir_vector, dim=-1, keepdim=True) + 1e-6
        )  # normalize
        signs = self.net(dir_vector).unsqueeze(-1)
        return signs


class NeuralFlowModel(nn.Module):
    def __init__(
        self,
        dim=3,
        latent_size=1,
        f_nlayers=4,
        f_width=50,
        s_nlayers=3,
        s_width=20,
        nonlinearity="relu",
        conformal=False,
        arch="imnet",
        no_sign_net=False,
        divfree=False,
        symm_dim=None,
    ):
        super(NeuralFlowModel, self).__init__()
        if conformal:
            model = ConformalDeformationFlowNetwork

        else:
            model = DeformationFlowNetwork
        self.no_sign_net = no_sign_net
        self.flow_net = model(
            dim=dim,
            latent_size=latent_size,
            nlayers=f_nlayers,
            width=f_width,
            nonlinearity=nonlinearity,
            arch=arch,
            divfree=divfree,
        )
        if not no_sign_net:
            self.sign_net = DeformationSignNetwork(
                latent_size=latent_size, nlayers=s_nlayers, width=s_width
            )
        self.symm_dim = symm_dim
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
        assert self.lat_params is not None
        return self.lat_params[idx]

    def update_latents(self, latent_sequence):
        """
        Args:
            latent_sequence: long or float tensor of shape
                [batch, nsteps, latent_size].
                sequence of latents along deformation path.
                if long, index into self.lat_params to retrieve latents.
        Returns:
            latent_waypoint: float tensor of shape [batch, nsteps], interp
                coefficient betwene [0, 1] corresponding to each latent code.
        """
        bs, ns, d = latent_sequence.shape
        dev = latent_sequence.device
        self.latent_sequence = latent_sequence
        self.latent_seq_len = torch.norm(
            self.latent_sequence[:, 1:] - self.latent_sequence[:, :-1], dim=-1
        )  # [batch, nsteps-1]
        self.latent_seq_len_sum = torch.sum(
            self.latent_seq_len, dim=1
        )  # [batch]
        self.latent_seq_weight = (
            self.latent_seq_len / self.latent_seq_len_sum[:, None]
        )  # [batch, nsteps-1]
        self.latent_seq_bins = torch.cumsum(
            self.latent_seq_weight, dim=1
        )  # [batch, nsteps-1]
        self.latent_seq_bins = torch.cat(
            [torch.zeros([bs, 1], device=dev), self.latent_seq_bins], dim=1
        )  # [batch, nsteps]
        self.latent_updated = True

        return self.latent_seq_bins

    def latent_at_t(self, t, return_sign=False):
        """Helper fn to compute latent at t."""
        t = t.to(self.latent_seq_bins.device)
        # find out which bin this t falls into
        bin_mask = (t > self.latent_seq_bins[:, :-1]) * (
            t < self.latent_seq_bins[:, 1:]
        )
        # logical and

        bin_mask = bin_mask.float()
        bin_idx = torch.argmax(bin_mask, dim=1)  # [batch,]
        batch_idx = torch.arange(bin_idx.shape[0]).to(bin_idx.device)

        # Find the interpolation coefficient between the latents at the two
        # ends of the bin
        t0 = self.latent_seq_bins[batch_idx, bin_idx]
        t1 = self.latent_seq_bins[batch_idx, bin_idx + 1]  # [batch]
        alpha = (t - t0) / (t1 - t0)  # [batch]
        latent_t0 = self.latent_sequence[
            batch_idx, bin_idx
        ]  # [batch, latent_size]
        latent_t1 = self.latent_sequence[
            batch_idx, bin_idx + 1
        ]  # [batch, latent_size]
        latent_val = latent_t0 + alpha[:, None] * (latent_t1 - latent_t0)
        latent_dir = (latent_t1 - latent_t0) / torch.norm(
            latent_t1 - latent_t0, dim=1, keepdim=True
        )
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
        # Reparametrize eval along latent path as a function of a single
        # scalar t
        if not self.latent_updated:
            raise RuntimeError(
                "Latent not updated. "
                "Use .update_latents() to update the source and target latents"
            )

        latent_val, latent_dir, sign = self.latent_at_t(t)
        sign = sign[:, None, None] * self.scale
        if self.symm_dim is None:
            flow = self.flow_net(latent_val, points)  # [batch, num_pints, dim]
        else:
            flow = symmetrize(self.flow_net, latent_val, points, self.symm_dim)
        # Normalize velocity based on time space proportional to latent
        # difference.
        flow *= self.latent_seq_len_sum[:, None, None]
        if not self.no_sign_net:
            sign = self.sign_net(latent_dir)
        return flow * sign


class NeuralFlowDeformer(nn.Module):
    def __init__(
        self,
        dim=3,
        latent_size=1,
        f_nlayers=4,
        f_width=50,
        s_nlayers=3,
        s_width=20,
        method="dopri5",
        nonlinearity="leakyrelu",
        arch="imnet",
        conformal=False,
        adjoint=True,
        atol=1e-5,
        rtol=1e-5,
        via_hub=False,
        no_sign_net=False,
        return_waypoints=False,
        use_latent_waypoints=False,
        divfree=False,
        symm_dim=None,
    ):
        """Initialize. The parameters are the parameters for the Deformation
        Flow network.

        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          f_nlayers: int, number of neural network layers for flow network.
            (>= 2).
          f_width: int, number of neurons per hidden layer for flow network
            (>= 1).
          s_nlayers: int, number of neural network layers for sign network
            (>= 2).
          s_width: int, number of neurons per hidden layer for sign network.
            (>= 1).
          arch: str, architecture, choice of 'imnet' / 'vanilla'
          adjoint: bool, whether to use adjoint solver to backprop gadient
            thru odeint.
          rtol, atol: float, relative / absolute error tolerence in ode solver.
          via_hub: bool, will perform transformation via hub-and-spokes
            configuration. Only useful if latent_sequence is torch.long
          return_waypoints: bool, return intermediate waypoints along timing.
          use_latent_waypoints: bool, use latent waypoints.
          symm_dim: int, list of int, or None. Symmetry axis/axes, or None.
        """
        super(NeuralFlowDeformer, self).__init__()
        self.method = method
        self.conformal = conformal
        self.arch = arch
        self.adjoint = adjoint
        self.odeint = odeint_adjoint if adjoint else odeint_regular
        self.__timing = torch.from_numpy(
            np.array([0.0, 1.0]).astype("float32")
        )
        self.return_waypoints = return_waypoints
        self.use_latent_waypoints = use_latent_waypoints
        self.rtol = rtol
        self.atol = atol
        self.via_hub = via_hub
        self.symm_dim = symm_dim

        self.net = NeuralFlowModel(
            dim=dim,
            latent_size=latent_size,
            f_nlayers=f_nlayers,
            f_width=f_width,
            s_nlayers=s_nlayers,
            s_width=s_width,
            arch=arch,
            conformal=conformal,
            nonlinearity=nonlinearity,
            no_sign_net=no_sign_net,
            divfree=divfree,
            symm_dim=symm_dim,
        )
        if symm_dim is not None:
            if not (isinstance(symm_dim, int) or isinstance(symm_dim, list)):
                raise ValueError(
                    "symm_dim must be int or list of ints, indicating axes of"
                    "symmetry."
                )

    @property
    def adjoint(self):
        return self.__adjoint

    @adjoint.setter
    def adjoint(self, isadjoint):
        assert isinstance(isadjoint, bool)
        self.__adjoint = isadjoint
        self.odeint = odeint_adjoint if isadjoint else odeint_regular

    @property
    def timing(self):
        return self.__timing

    @timing.setter
    def timing(self, timing):
        assert isinstance(timing, torch.Tensor)
        assert timing.ndim == 1
        self.__timing = timing

    def add_encoder(self, encoder):
        self.net.add_encoder(encoder)

    def add_lat_params(self, lat_params):
        self.net.add_lat_params(lat_params)

    def get_lat_params(self, idx):
        return self.net.get_lat_params(idx)

    def forward(self, points, latent_sequence):
        """Forward transformation (source -> latent_path -> target).

        To perform backward transformation, simply switch the order of the lat
        codes.

        Args:
          points: [batch, num_points, dim]
          latent_sequence: float tensor of shape [batch, nsteps, latent_size],
            ------- or -------
            long tensor of shape [batch, nsteps]
            sequence of latents along deformation path.
            if long, index into self.lat_params to retrieve latents.
        Returns:
          points_transformed:
            tensor of shape [batch, num_points, dim] if not
            self.return_waypoint.
            tensor of shape [nsteps, batch, num_points, dim] if
            self.return_waypoint.
        """
        if latent_sequence.dtype == torch.long:
            latent_sequence = self.get_lat_params(
                latent_sequence
            )  # [nsteps, batch, lat_dim]
            if self.via_hub:
                assert latent_sequence.shape[1] == 2
                zeros = torch.zeros_like(latent_sequence[:, :1])
                lat0 = latent_sequence[:, 0:1]
                lat1 = latent_sequence[:, 1:2]
                latent_sequence = torch.cat(
                    [lat0, zeros, lat1], dim=1
                )  # [batch, nsteps=3, lat_dim]
        waypoints = self.net.update_latents(latent_sequence)
        if self.use_latent_waypoints:
            timing = waypoints[0]
        else:
            timing = self.timing
        points_transformed = self.odeint(
            self.net,
            points,
            timing,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )
        if self.return_waypoints:
            return points_transformed
        else:
            return points_transformed[-1]
