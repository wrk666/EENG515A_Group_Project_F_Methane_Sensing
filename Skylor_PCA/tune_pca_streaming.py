import copy 
import csv
import numpy as np

from main import (
    run, make_training_flights, IdentityDenoiser, WindowPCADenoiser
)

# Nine combos to compare
windowLengths = [21, 41, 61]
componentCounts = [4, 9, 15]

# Seperate noise realizations for selection and final evaluation
validationSeed = [202, 303, 404]
testSeeds = [505, 606, 707]

print('Program is currently running...', flush=True)

train = make_training_flights(seed=1)

def evaluate(denoiser, seed):
    _, log = run('lawnmower', seed=seed, denoiser=denoiser)

    mse = np.mean((log['y'] - log['y_clean']) ** 2)
    rmse = np.sqrt(mse)

    emissionError = 100 * abs(log['q_neigh_rel_err'][-1])

    return rmse, emissionError

# save ito file
with open('pca_tuning_results.csv', 'w', newline='') as file:
    write = csv.writer(file)
    write.writerow([
        'stage', 'method', 'window', 'components', 'seed',
        'concentration_rmse_ppm', 'emission_error_percent'])

    def record(stage, method, window, components, seed, results):
        rmse, error = results
        write.writerow([
            stage, method, window, components, seed, rmse, error
        ])
        file.flush()

    # Raw measurements provide the validation baseline
    for seed in validationSeed:
        results = evaluate(IdentityDenoiser(), seed)
        record('validation', 'Raw', 0, 0, seed, results)

    summaries = []

    #compare pca combinations on validation runs
    for window in windowLengths:
        for components in componentCounts:

            fitted = WindowPCADenoiser(
                train, window=window, n_components=components)

            errors = []
            rmses = []

            for seed in validationSeed:
                #Fresh history for each flight; reuse the fitted PCA
                denoiser = copy.deepcopy(fitted)
                results = evaluate(denoiser, seed)

                record(
                    'validation', 'PCA',
                    window, components, seed, results)

                rmses.append(results[0])
                errors.append(results[1])

            summaries.append((
                np.mean(errors), np.mean(rmses), window, components))

    #rank by average validation emission error
    summaries.sort(key=lambda row: row[0])

    _, _, bestWindow, bestComp = summaries[0]

    #   Evaluate only the selected combination on fresh test noise
    selected = WindowPCADenoiser(
        train, window=bestWindow, n_components=bestComp)

    test_errors = {'Raw': [], 'PCA': []}
    test_rmses = {'Raw': [], 'PCA': []}

    for seed in testSeeds:
        for method in ['Raw', 'PCA']:
            if method == 'Raw':
                denoiser = IdentityDenoiser()
                window, components = 0, 0
            else:
                denoiser = copy.deepcopy(selected)
                window, components = bestWindow, bestComp

            results = evaluate(denoiser, seed)
            record('test', method, window, components, seed, results)

            test_rmses[method].append(results[0])
            test_errors[method].append(results[1])

print('All done, results can be seen in pca_tuning_results.csv')