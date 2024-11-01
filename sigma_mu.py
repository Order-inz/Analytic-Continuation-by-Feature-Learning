import random
import csv
import os

# number of Gaussian samples
SAMPLE_N = 100000
# maximum number of peaks, smaller number of peaks generation is favored
PEAK_N = 8
# weights_number = int(xi**BIAS_PEAK_N*PEAK_N+1), BIAS_PEAK_N > 1 means samples
# with smaller number of peaks has more generations
BIAS_PEAK_N = 1.5


def weights(weights_num):
    xis = [0.0, 1.0]
    for i in range(weights_num - 1):
        xis.append(random.random())
    xis.sort()
    weightList = []
    for i in range(weights_num):
        weightList.append(xis[i + 1] - xis[i])
    return weightList


def gauss_list(sample_n):
    gauss_list = []
    mu_sigma_list = []

    for isample in range(sample_n):
        xi = random.random()
        peakN = int(xi ** BIAS_PEAK_N * PEAK_N + 1)
        weightList = weights(peakN)
        sigmaList = []
        muList = []
        mu_sigma = []
        for i in range(peakN):
            mu = 10*random.random() - 5
            sigma = 3.9 * random.random()**2+0.1

            muList.append(mu)
            sigmaList.append(sigma)

            mu_sigma.extend([mu, sigma])

        mu_sigma_list.append(mu_sigma)
        gauss_list.append({"peakN": peakN, "weightList": weightList, "sigmaList": sigmaList, "muList": muList})

    return gauss_list, mu_sigma_list


def save_gauss_list(gauss_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['peakN', 'weightList', 'sigmaList', 'muList']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in gauss_list:
            writer.writerow(data)


def save_mu_sigma(filename, mu_sigma_list):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for mu_sigma in mu_sigma_list:
            writer.writerow(mu_sigma)


if __name__ == "__main__":
    gauss_list, mu_sigma_list = gauss_list(SAMPLE_N)
    save_gauss_list(gauss_list, 'gauss_list.csv')
    save_mu_sigma('mu_sigma.csv', mu_sigma_list)



