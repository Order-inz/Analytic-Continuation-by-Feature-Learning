import csv
import numpy as np
import scipy.integrate as integrate

# inverse temperature
BETA = 10.0
# number of positive/negative imaginary frequencies to be retained [-2*n+1, 2*n-1]
OMEGA_POINTS_N = 32
# number of Gaussian samples
SAMPLE_N = 100000
# minimum and maximum positions of the the Gaussian peaks
OMEGA_MIN = -10.0
OMEGA_MAX = 10.0
EPS = 1E-10


def load_gauss_list(file_name):
    gauss_list = []
    with open(file_name, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gauss_list.append({
                'peakN': int(row['peakN']),
                'weightList': [float(x) for x in row['weightList'][1:-1].split(',')],
                'sigmaList': [float(x) for x in row['sigmaList'][1:-1].split(',')],
                'muList': [float(x) for x in row['muList'][1:-1].split(',')]
            })
    return gauss_list


def gauss_function(x, weight, sigma, mu):
    factor = 1.0 / np.sqrt(2.0 * np.pi)
    return factor * weight * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / sigma


def green_function(weight, omega_n, sigma, mu):
    factor = 1.0 / np.sqrt(2.0 * np.pi)
    real = lambda x: - weight*(mu+x*sigma) / (omega_n**2 + (mu+x*sigma)**2) * np.exp(-x**2/2.0)
    imag = lambda x: - weight*omega_n / (omega_n**2 + (mu+x*sigma)**2) * np.exp(-x**2/2.0)

    real_g = integrate.quad(real, OMEGA_MIN, OMEGA_MAX, epsabs=EPS, epsrel=0)
    imag_g = integrate.quad(imag, OMEGA_MIN, OMEGA_MAX, epsabs=EPS, epsrel=0)
    real_g = (real_g[0]*factor, real_g[1]*factor)
    imag_g = (imag_g[0]*factor, imag_g[1]*factor)

    return real_g, imag_g


def compute_gauss_and_green(gauss_list):
    gauss_results = []
    green_results = []

    for data in gauss_list:
        peakN = data['peakN']
        weightList = data['weightList']
        sigmaList = data['sigmaList']
        muList = data['muList']

        x_values = np.linspace(OMEGA_MIN, OMEGA_MAX, 256)
        total_gauss = np.zeros_like(x_values)
        total_green = np.zeros(2 * OMEGA_POINTS_N, dtype=complex)

        for i in range(peakN):
            weight = weightList[i]
            sigma = sigmaList[i]
            mu = muList[i]

            gauss_fn_values = gauss_function(x_values, weight, sigma, mu)
            total_gauss += gauss_fn_values

        for n in range(-OMEGA_POINTS_N, OMEGA_POINTS_N):
            omega_n = (2 * n + 1) * np.pi / BETA
            for i in range(peakN):
                weight = weightList[i]
                sigma = sigmaList[i]
                mu = muList[i]

                real_green, imag_green = green_function(weight, omega_n, sigma, mu)
                total_green[n + OMEGA_POINTS_N] += complex(real_green[0], imag_green[0])

        gauss_results.append(total_gauss)
        green_results.append(total_green)

    return gauss_results, green_results


def save_gauss_functions(gauss_results, file_name):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for gauss_fn in gauss_results:
            writer.writerow(gauss_fn.tolist())


def save_green_functions(green_results, file_name):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for green_fn in green_results:
            real_green = green_fn.real.tolist()
            imag_green = green_fn.imag.tolist()
            row = []
            for real, imag in zip(real_green, imag_green):
                row.append(real)
                row.append(imag)
            writer.writerow(row)


def main():
    gauss_list = load_gauss_list('gauss_list.csv')
    gauss_results, green_results = compute_gauss_and_green(gauss_list)

    save_gauss_functions(gauss_results, '8_A.csv')
    save_green_functions(green_results, '8_G.csv')


if __name__ == "__main__":
    main()