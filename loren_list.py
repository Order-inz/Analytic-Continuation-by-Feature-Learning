import random
import csv

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


def loren_list(sample_n):
    loren_list = []

    for isample in range(sample_n):
        xi = random.random()
        peakN = int(xi ** BIAS_PEAK_N * PEAK_N + 1)
        weightList = weights(peakN)
        gamaList = []
        x0List = []
        for i in range(peakN):
            gamaList.append(2.9*random.random()+0.1)
            x0List.append(10*random.random()-5)

        loren_list.append({"peakN": peakN, "weightList": weightList, "gamaList": gamaList, "x0List": x0List})

    return loren_list


def save_loren_list(loren_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['peakN', 'weightList', 'gamaList', 'x0List']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in loren_list:
            writer.writerow(data)


if __name__ == "__main__":
    loren_list = loren_list(SAMPLE_N)
    save_loren_list(loren_list, 'loren_list.csv')
