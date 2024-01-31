#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# Z |-> G_\theta(Z)
############################################################################
from torchdyn.core import NeuralODE
import torch
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from sklearn.preprocessing import StandardScaler
import numpy as np


class MLP(torch.nn.Module):
    def __init__(self, dim, cond_dim=0, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + cond_dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )
        self.cond_dim = cond_dim

    def forward(self, x):
        return torch.cat((self.net(x), torch.zeros((x.shape[0], self.cond_dim))), dim=1)

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim=4)
        input noise of the generative model
    """
    # See below an example
    # ---------------------
    latent_variable = noise[:, :28]  # choose the appropriate latent dimension of your model

    # load your parameters or your model
    # <!> be sure that they are stored in the parameters/ directory <!>
    model = MLP(dim=28, w=44, time_varying=True)
    model.load_state_dict(torch.load('parameters/OptimalTransportCFM'))

    scale = np.array([1.79011101, 1.25945766, 1.92629044, 1.67930392, 1.10472514, 0.84796846, 1.08745545, 1.16562508, 1.14689738, 1.12532645, 
                   1.41768747, 1.4007612 , 1.4810414 , 1.1969566 , 1.18712546, 1.09597999, 1.0677837 , 0.42060002, 1.26409699, 0.5948676 ,
                   0.25145016, 0.62057106, 0.83855863, 0.45740406, 0.89509364, 0.65312771, 0.33486501, 0.61451865])

    mean = np.array([ 9.38902   ,  5.36495   ,  3.31693   ,  6.16897   , 22.44082138, 26.93509719, 23.93751689, 21.72991963, 27.22071612, 21.70814759,
                      17.28449936, 22.68562893, 17.41391925, 19.55009987, 24.65488531, 19.06652853,  3.25862371,  1.48305166,  3.54321322,  1.62085769,
                      0.66964528,  1.55133938,  2.74239638,  1.64320753,  2.29387879, 1.79292258,  1.16785829,  1.63855951])

    nb_samples = latent_variable.shape[0]
    with torch.no_grad():
        node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        x0 = torch.rand([nb_samples,28], dtype=torch.float32)
        traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, 100),
                )
        samples = (scale*traj[-1].cpu().numpy() + mean) #Inverse Scaler
        samples = samples[:,:4]
    print(samples)
    return samples 