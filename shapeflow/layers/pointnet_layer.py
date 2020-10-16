"""
Credits:
Originally implemented by Fei Xia.
https://github.com/fxia22/pointnet.pytorch

with small modifications.
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np

from .shared_definition import NORMTYPE, NONLINEARITIES


class STN3d(nn.Module):
    def __init__(self, norm_type, nonlinearity="relu"):
        super(STN3d, self).__init__()
        assert norm_type in NORMTYPE.keys()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = torch.nn.Conv1d(1024, 512, 1)
        self.fc2 = torch.nn.Conv1d(512, 256, 1)
        self.fc3 = torch.nn.Conv1d(256, 9, 1)
        self.nl = NONLINEARITIES[nonlinearity]

        self.bn1 = NORMTYPE[norm_type](64)
        self.bn2 = NORMTYPE[norm_type](128)
        self.bn3 = NORMTYPE[norm_type](1024)
        self.bn4 = NORMTYPE[norm_type](512)
        self.bn5 = NORMTYPE[norm_type](256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.nl(self.bn1(self.conv1(x)))
        x = self.nl(self.bn2(self.conv2(x)))
        x = self.nl(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]  # [b, c, 1]

        x = self.nl(self.bn4(self.fc1(x)))  # [b, c, 1]
        x = self.nl(self.bn5(self.fc2(x)))  # [b, c, 1]
        x = self.fc3(x).squeeze(-1)

        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(
        self,
        in_features=3,
        nf=64,
        global_feat=True,
        feature_transform=False,
        norm_type="batchnorm",
        nonlinearity="relu",
    ):
        super(PointNetfeat, self).__init__()
        assert norm_type in NORMTYPE.keys()
        self.stn = STN3d(norm_type=norm_type, nonlinearity=nonlinearity)
        self.conv1 = torch.nn.Conv1d(in_features, nf, 1)
        self.conv2 = torch.nn.Conv1d(nf, nf * 2, 1)
        self.conv3 = torch.nn.Conv1d(nf * 2, nf * 16, 1)
        self.nm1 = NORMTYPE[norm_type](nf)
        self.nm2 = NORMTYPE[norm_type](nf * 2)
        self.nm3 = NORMTYPE[norm_type](nf * 16)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.nf = nf
        self.nl = NONLINEARITIES[nonlinearity]
        if self.feature_transform:
            self.fstn = STN3d(k=nf)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.nl(self.nm1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.nl(self.nm2(self.conv2(x)))
        x = self.nm3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.nf * 16)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.nf * 16, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetEncoder(nn.Module):
    def __init__(
        self,
        nf=64,
        in_features=3,
        out_features=8,
        feature_transform=False,
        dropout_prob=0.3,
        norm_type="batchnorm",
        nonlinearity="relu",
    ):
        super(PointNetEncoder, self).__init__()
        assert norm_type in NORMTYPE.keys()
        self.feature_transform = feature_transform
        self.dropout_prob = dropout_prob
        self.feat = PointNetfeat(
            global_feat=True,
            in_features=in_features,
            nf=nf,
            feature_transform=feature_transform,
            norm_type=norm_type,
            nonlinearity=nonlinearity,
        )
        self.fc1 = nn.Conv1d(nf * 16, nf * 8, 1)
        self.fc2 = nn.Conv1d(nf * 8, nf * 4, 1)
        self.fc3 = nn.Conv1d(nf * 4, out_features, 1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.nm1 = NORMTYPE[norm_type](nf * 8)
        self.nm2 = NORMTYPE[norm_type](nf * 4)
        self.nl = NONLINEARITIES[nonlinearity]
        self.in_features = in_features
        self.out_features = out_features
        self.nf = nf

    def forward(self, x):
        """
        Args:
          x: tensor of shape [batch, npoints, in_features]
        Returns:
          output: tensor of shape [batch, out_features]
        """
        x = x.permute(0, 2, 1)
        x, _, _ = self.feat(x)
        x = x.unsqueeze(-1)
        x = self.nl(self.nm1(self.fc1(x)))
        x = self.nl(
            self.nm2(self.dropout(self.fc2(x).squeeze(-1)).unsqueeze(-1))
        )
        x = self.fc3(x).squeeze(-1)
        return x


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]  # noqa: E741
    if trans.is_cuda:
        I = I.cuda()  # noqa: E741
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss


if __name__ == "__main__":
    # example for using encoder
    encoder = PointNetEncoder(output_features=8)
    points = torch.rand(16, 100, 3)
    latents = encoder(points)
    print(latents.shape)
