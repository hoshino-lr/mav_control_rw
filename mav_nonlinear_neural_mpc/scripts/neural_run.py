import numpy
import numpy as np
import matplotlib.pyplot as plt
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_tensor_type('torch.DoubleTensor')

 
 
# choose_device
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.cuda.set_device(0)

Model = collections.namedtuple('Model', 'phi h options')


class Phi_Net(nn.Module):
    def __init__(self, options):
        super(Phi_Net, self).__init__()

        self.fc1 = nn.Linear(options['dim_x'], 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 50)
        # One of the NN outputs is a constant bias term, which is append below
        self.fc4 = nn.Linear(50, options['dim_a'] - 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if len(x.shape) == 1:
            # single input
            return torch.cat([x, torch.ones(1)])
        else:
            # batch input for training
            return torch.cat([x, torch.ones([x.shape[0], 1])], dim=-1)


def load_model(modelname):
    model = torch.load(modelname + '.pth')
    options = model['options']

    phi_net = Phi_Net(options=options)
    h_net = None

    phi_net.load_state_dict(model['phi_net_state_dict'])

    phi_net.eval()

    return Model(phi_net, h_net, options)


_softmax = nn.Softmax(dim=1)

net_file = "./net_resources/net_915"

model = load_model(net_file)

options = {'dim_a': 3, 'loss_type': 'crossentropy-loss'}
phi_net = model.phi
h_net = model.h
lam = 0
# v  q(x y z w)  pwm / 1000
# input 1 x 11
# output 1 x 3
valid_inputs = numpy.ndarray([1, 11])

inputs = torch.from_numpy(valid_inputs)

with torch.no_grad():
    # Compute adversarial network prediction
    output_phi = phi_net(inputs)

