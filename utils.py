import torch
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import torch.nn as nn
from ot.sliced import sliced_wasserstein_distance

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))

def D_train(latent_dim, x, G, D, D_optimizer, device, distr="normal", log=None):
    z = sample_from(distr, x.shape[0], latent_dim, device)
    fake_batch = G(z)
    D_scores_on_real = D(x.to(device))
    D_scores_on_fake = D(fake_batch)
    
    # alternative computation - might yield better resuls
    # change G accordingly if used
    # D_loss = -torch.mean(torch.log(D_scores_on_fake) + torch.log(1 - D_scores_on_real))

    y_real, y_fake = torch.ones(x.shape[0], 1).to(device), torch.zeros(x.shape[0], 1).to(device)
    D_real_loss = nn.BCELoss()(D_scores_on_real, y_real)
    D_fake_loss = nn.BCELoss()(D_scores_on_fake, y_fake)
    D_loss = - (D_real_loss + D_fake_loss)

    D_optimizer.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    # log to tensorboard
    # if log is not None:
    #     discriminator_log(log, D_scores_on_real.mean(), D_scores_on_fake.mean(), D_loss)

    return D_loss.data.item()

def G_train(latent_dim, x, G, D, G_optimizer, device, distr="normal", log=None):
    z = sample_from(distr, x.shape[0], latent_dim, device)
    fake_batch = G(z)
    D_scores_on_fake = D(fake_batch)
    
    # alternative computation - might yield better resuls
    # change D accordingly if used
    # G_loss = -torch.mean(torch.log(1 - D_scores_on_fake))

    y = torch.ones(x.shape[0], 1).to(device)
    G_loss = nn.BCELoss()(D_scores_on_fake, y)

    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    # log to tensorboard
    # if log is not None:
    #     generator_log(log, x, fake_batch, G_loss)

    return G_loss.item()

def sample_from(distr, n_samples, latent_dim, device):
    if distr == "normal":
        return torch.randn(n_samples, latent_dim).to(device)
    elif distr == "exp":
        return torch.Tensor.exponential_(torch.zeros((n_samples, latent_dim)), 2).to(device)
    elif distr == "gamma":
        return torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([1.0])).sample((n_samples, latent_dim))[:,:,0].to(device)
    elif distr == "uniform":
        return torch.rand(n_samples, latent_dim).to(device)
    elif distr == "student":
        return torch.distributions.studentT.StudentT(torch.tensor([2.0])).sample((n_samples, latent_dim))[:,:,0].to(device)

def D_wasserstrain(latent_dim, x, G, D, D_optimizer, device, distr="normal", log=None):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x
    x_real = x_real.to(device)

    D_output_real = D(x_real)

    # train discriminator on fake
    z = sample_from(distr, x.shape[0], latent_dim, device)
    x_fake = G(z)

    D_output_fake = D(x_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = (D_output_real - D_output_fake).mean()
    D_loss.backward()
    D_optimizer.step()

    for p in D.parameters():
        p.data = torch.clamp(p.data, -0.01, 0.01)

    # log to tensorboard
    # if log is not None:
    #     discriminator_log(log, D_output_real.mean(), D_output_fake.mean(), D_loss)

    return D_loss.data.item()

def G_wasserstrain(latent_dim, x, G, D, G_optimizer, device, distr="normal", log=None):
    # =======================Train the generator=======================#
    G.zero_grad()

    z = sample_from(distr, x.shape[0], latent_dim, device)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = - D_output.mean()

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    # log to tensorboard
    # if log is not None:
    #     generator_log(log, x, G_output, G_loss)

    return G_loss.data.item()

def energy_distance(x, y):
    n=x.shape[0]
    m=y.shape[0]

    x_reshaped1 = torch.unsqueeze(x, 1)
    y_reshaped2 = torch.unsqueeze(y, 0)
    x_reshaped2 = torch.unsqueeze(x, 0)
    y_reshaped1 = torch.unsqueeze(y, 1)

    normsA = torch.linalg.norm(x_reshaped1 - y_reshaped2, axis=2) 
    A = torch.sum(normsA)
    normsB = torch.linalg.norm(x_reshaped1 - x_reshaped2, axis=2) 
    B = torch.sum(normsB) 
    normsC = torch.linalg.norm(y_reshaped1 - y_reshaped2, axis=2) 
    C = torch.sum(normsC)

    result = 2*A/(n*m) - C/(m**2)  - B/(n**2) 
    return result

# Attempt at Energy Distance model
def ED_model_step(latent_dim, x, G, G_optimizer, device, distr="normal", log=None):

    G.zero_grad()

    z = sample_from(distr, x.shape[0], latent_dim, device)

    G_output = G(z)

    G_loss = energy_distance(x.to(device), G_output)

    G_loss.backward()

    G_optimizer.step()

    if log is not None:
        generator_log(log, x.to(device), G_output, G_loss)

    return G_loss.data.item()

def make_fake_data(distr, n, latent_dim, G):
    z = sample_from(distr, n, latent_dim, "cpu")
    fake_samples = G(z)
    fake_data = fake_samples.cpu().data.numpy()
    return fake_data

def make_fake_data_renorm(distr, n, latent_dim, G, means, stds):
    z = sample_from(distr, n, latent_dim, "cpu")
    fake_samples = G(z)
    fake_data = fake_samples.cpu().data.numpy()
    fake_data = fake_data * np.array(stds.cpu()) + np.array(means.cpu())
    return fake_data

# def plot_fake_data(fake_data, log, folder="ecdfs/"):
#     fig, ax = plt.subplots(figsize = (12,6))
    
#     for k in range(fake_data.shape[1]):
#         yield_ind = X.columns[k]
#         plt.subplot(4, 4, 5*k + 1)

#         samples[k].plot.density()
#         X[yield_ind].plot.density()

#         xmin, xmax = np.min(X[yield_ind]), np.max(X[yield_ind])
#         plt.xlim(xmin, xmax)

#         plt.title(yield_ind + " histogram")
#         plt.xlabel("Crop yields in (hundres of tons ?)")
#         plt.ylabel("Density")
#         plt.legend()

#     # Plotting scatter plots for correlation
#     for i in range(samples.shape[1]):
#         for j in range(samples.shape[1]):
#             if i != j:
#                 plt.subplot(4, 4, 1 + i + 4*j)

#                 x = samples.iloc[:, i]
#                 y = samples.iloc[:, j]
#                 plt.scatter(x, y, alpha=0.5)

#                 x = X.iloc[:, i]
#                 y = X.iloc[:, j]
#                 plt.scatter(x, y, alpha=0.5)

#                 plt.title(f"{X.columns[i]} vs {X.columns[j]}")
#                 plt.xlabel(X.columns[i])
#                 plt.ylabel(X.columns[j])

#     plt.suptitle("Fitting the marginal histogram and Scatter plots for Correlation")
#     plt.tight_layout()
#     plt.show()

#     ax.set_xlabel('Value')
#     ax.set_ylabel('Empirical CDF')
#     ax.legend(["X1", "X2", "X3", "X4"])

#     writer, idx, ep, num_batches = log
#     writer.add_figure(folder + "{:04}".format(idx), fig,
#                             global_step=ep * num_batches + idx)
#     plt.close(fig)

def metrics_log(original_data, fake_data, log, folder="metrics/"):
    writer, idx, ep, num_batches = log
    writer.add_scalar(folder + "sliced_wasserstein", sliced_wasserstein_distance(original_data, fake_data, n_projections=1000),
                      global_step=ep * num_batches + idx)
    
def metrics_log_train(original_data, fake_data, log, folder="metrics/"):
    writer, idx, ep, num_batches = log
    writer.add_scalar(folder + "sliced_wasserstein_train", sliced_wasserstein_distance(original_data, fake_data, n_projections=1000),
                      global_step=ep * num_batches + idx)
    
def metrics_log_test(original_data, fake_data, log, folder="metrics/"):
    writer, idx, ep, num_batches = log
    writer.add_scalar(folder + "sliced_wasserstein_test", sliced_wasserstein_distance(original_data, fake_data, n_projections=1000),
                      global_step=ep * num_batches + idx)
