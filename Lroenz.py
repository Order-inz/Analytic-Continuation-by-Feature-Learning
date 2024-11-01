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


def load_loren_list(file_name):
    loren_list = []
    with open(file_name, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            loren_list.append({
                'peakN': int(row['peakN']),
                'weightList': [float(x) for x in row['weightList'][1:-1].split(',')],
                'gamaList': [float(x) for x in row['gamaList'][1:-1].split(',')],
                'x0List': [float(x) for x in row['x0List'][1:-1].split(',')]
            })
    return loren_list


def loren_function(x, weight, gama, x0):
    loren = weight * gama / (np.pi * ((x - x0) ** 2 + gama ** 2))
    return loren


def green_function(weight, omega_n, gama, x0):
    real = lambda x: - weight * (x + x0) * gama / (omega_n**2 + (x + x0)**2) / (np.pi * (x ** 2 + gama ** 2))
    imag = lambda x: - weight * omega_n * gama / (omega_n**2 + (x + x0)**2) / (np.pi * (x ** 2 + gama ** 2))

    real_g = integrate.quad(real, OMEGA_MIN, OMEGA_MAX, epsabs=EPS, epsrel=0)
    imag_g = integrate.quad(imag, OMEGA_MIN, OMEGA_MAX, epsabs=EPS, epsrel=0)
    real_g = (real_g[0], real_g[1])
    imag_g = (imag_g[0], imag_g[1])

    return real_g, imag_g


def compute_loren_and_green(loren_list):
    loren_results = []
    green_results = []

    for data in loren_list:
        peakN = data['peakN']
        weightList = data['weightList']
        gamaList = data['gamaList']
        x0List = data['x0List']

        x_values = np.linspace(OMEGA_MIN, OMEGA_MAX, 256)
        total_loren = np.zeros_like(x_values)
        total_green = np.zeros(2 * OMEGA_POINTS_N, dtype=complex)

        for i in range(peakN):
            weight = weightList[i]
            gama = gamaList[i]
            x0 = x0List[i]

            loren_fn_values = loren_function(x_values, weight, gama, x0)
            total_loren += loren_fn_values

        for n in range(-OMEGA_POINTS_N, OMEGA_POINTS_N):
            omega_n = (2 * n + 1) * np.pi / BETA
            for i in range(peakN):
                weight = weightList[i]
                gama = gamaList[i]
                x0 = x0List[i]

                real_green, imag_green = green_function(weight, omega_n, gama, x0)
                total_green[n + OMEGA_POINTS_N] += complex(real_green[0], imag_green[0])

        loren_results.append(total_loren)
        green_results.append(total_green)

    return loren_results, green_results


def save_loren_functions(loren_results, file_name):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for loren_fn in loren_results:
            writer.writerow(loren_fn.tolist())


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
    loren_list = load_loren_list('loren_list.csv')
    loren_results, green_results = compute_loren_and_green(loren_list)

    save_loren_functions(loren_results, 'TEST_2l_A.csv')
    save_green_functions(green_results, 'TEST_2l_G.csv')


if __name__ == "__main__":
    main()