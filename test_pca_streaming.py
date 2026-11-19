import numpy as np

from main import (
    run, make_training_flights, IdentityDenoiser, WindowPCADenoiser
)

# Use same train data for each comparison
train = make_training_flights()

# Chekc three independent test-noise realizations
for seed in [0, 42, 100]:
    print('\nTest Seed:', seed, flush=True)

    for name in ['Raw', 'PCA']:
        if name == 'Raw':
            denoise = IdentityDenoiser()
        else:
            denoise = WindowPCADenoiser(train, window=21, n_components=9)

        # Using matching noise and the same fixed path for the two methods
        _, log = run('lawnmower', seed=seed, denoiser=denoise)

        # Mean squared error #
        mse = np.mean((log['y'] - log['y_clean']) ** 2)
        # Root mean squared error
        rmse = np.sqrt(mse)

        ### PRINT ###

        print(name, flush=True)
        print('Concentration RMSE (ppm):', rmse, flush=True)
        print('Neighborhood emission estimate (g/s):', log['q_neigh'][-1],
              flush=True)
        print('Relative emission error (%):', 100 * log['q_neigh_rel_err'][-1],
              flush=True)