import math
import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data_utils
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from ot.sliced import sliced_wasserstein_distance

from GAN import Generator, Discriminator
from utils import ED_model_step
from utils import D_train, G_train, save_models
from utils import D_wasserstrain, G_wasserstrain
from utils import make_fake_data, make_fake_data_renorm, metrics_log_train, metrics_log_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD.")
    parser.add_argument("--latent_dim", type=int, default=16, 
                        help="Latent space dimension.")
    parser.add_argument("--g_hidden_dim", type=int, default=64, 
                        help="Impacts generator number of parameters.")
    parser.add_argument("--d_hidden_dim", type=int, default=64, 
                        help="Impacts discriminator number of parameters.")
    parser.add_argument("--latent_distr", type=str, default="normal", 
                        help="Latent space dimension.")
    parser.add_argument("--joint", type=str, default=True, 
                        help="Whether to generate joint distribution.")
    parser.add_argument("--version", "-v", type=str, default="test",
                        help="Name of run for Tensorboard.")

    args = parser.parse_args()


    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    # Data Pipeline
    print('Dataset loading...')

    S40 = pd.read_csv("data/station_40.csv")
    S49 = pd.read_csv("data/station_49.csv")
    S63 = pd.read_csv("data/station_63.csv")
    S80 = pd.read_csv("data/station_80.csv")

    S40.rename(columns=lambda x: x + "_40" if x != "YEAR" else x, inplace=True)
    S49.rename(columns=lambda x: x + "_49" if x != "YEAR" else x, inplace=True)
    S63.rename(columns=lambda x: x + "_63" if x != "YEAR" else x, inplace=True)
    S80.rename(columns=lambda x: x + "_80" if x != "YEAR" else x, inplace=True)

    merged_df = pd.merge(S40, S49, on='YEAR', how='inner')
    merged_df = pd.merge(merged_df, S63, on='YEAR', how='inner')
    dataset = pd.merge(merged_df, S80, on='YEAR', how='inner')

    dataset.set_index('YEAR', inplace=True)
    target = ['YIELD_40', 'YIELD_49', 'YIELD_63', 'YIELD_80']
    features = [x for x in dataset.columns if x not in target]

    dataset['Q40'] = (dataset['W_13_40'] + dataset['W_14_40'] + dataset['W_15_40'])
    dataset['Q49'] = (dataset['W_13_49'] + dataset['W_14_49'] + dataset['W_15_49'])
    dataset['Q63'] = (dataset['W_13_63'] + dataset['W_14_63'] + dataset['W_15_63'])
    dataset['Q80'] = (dataset['W_13_80'] + dataset['W_14_80'] + dataset['W_15_80'])

    features = ['YIELD_40', 'YIELD_49', 'YIELD_63', 'YIELD_80']
    aux_columns = ['Q40', 'Q49', 'Q63', 'Q80']
    nb_weather_var = len([x for x in dataset.columns if not (x in features or x in aux_columns) ])

    dataset = dataset[(dataset['Q40'] <= 6.4897) & (dataset['Q49'] <= 3.3241) & (dataset['Q63'] <= 7.1301) & (dataset['Q80'] <= 5.1292)]
    dataset.drop(['Q40', 'Q49', 'Q63', 'Q80'], axis=1, inplace=True)

    #Train / Test split :
    train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)

    # Use only the YIELD data :
    X = train_data[features]
    X_train = X.copy()
    X_test = test_data[features]

    train = torch.tensor(X_train.values.astype(np.float32))
    train_dataset = data_utils.TensorDataset(train)
    train_loader = data_utils.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)

    # transform data before
    means = train.mean(dim=0, keepdim=True)
    stds = train.std(dim=0, keepdim=True)
    train_normalized = (train - means) / stds

    # data loaders
    train_dataset_normalized = data_utils.TensorDataset(train_normalized)
    train_loader_normalized = data_utils.DataLoader(dataset = train_dataset_normalized, batch_size = args.batch_size, shuffle = True)

    # define writer to accompany training
    writer = SummaryWriter(log_dir=f"tb_logs/{args.version}", purge_step=0)

    dim = 4
    G = torch.nn.DataParallel(Generator(latent_dim=args.latent_dim, g_hidden_dim=args.g_hidden_dim, g_output_dim=dim)).to(device)
    D = torch.nn.DataParallel(Discriminator(d_input_dim=dim, d_hidden_dim=args.d_hidden_dim)).to(device)

    G_optimizer = torch.optim.RMSprop(G.parameters(), lr=args.lr)
    D_optimizer = torch.optim.RMSprop(D.parameters(), lr=args.lr, maximize=True)

    d_losses, g_losses = [], []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        with tqdm(enumerate(train_loader_normalized), total=len(train_loader_normalized), leave=False, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, x in pbar:
                x = x[0]
                log = (writer, batch_idx, epoch, len(train_loader))
                d_loss = D_wasserstrain(args.latent_dim, x, G, D, D_optimizer, device, log=log)
                if batch_idx % 5 == 0:
                    g_loss = G_wasserstrain(args.latent_dim, x, G, D, G_optimizer, device, log=log)
                    d_losses.append(d_loss)
                    g_losses.append(g_loss)
        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')
            
            # # plot fake data
            # nb_samples = 1000
            # fake_data = make_fake_data_renorm(args.latent_distr, nb_samples, args.latent_dim, G, means, stds)
            # plot_fake_data(fake_data, log)

            fake_data_train = make_fake_data_renorm(args.latent_distr, X_train.shape[0], args.latent_dim, G, means, stds)
            fake_data_test = make_fake_data_renorm(args.latent_distr, X_test.shape[0], args.latent_dim, G, means, stds)
            metrics_log_train(X_train.values, fake_data_train, log=log)
            metrics_log_test(X_test.values, fake_data_test, log=log)