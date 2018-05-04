import matplotlib
matplotlib.use('Agg')
import sys
import csv
import codecs
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

def write_csv(name='output.csv', header=None, data=None):
    '''
    header: name of column. (ex) ["Name", "Accuracy", "Loss"]
    data: 2-dim array
    '''
    # if sys.version_info.major==2:
    with codecs.open(name, 'w', encoding='utf-8') as f:
        wr = csv.writer(f)
        if header is not None:
            wr.writerow(header)
        wr.writerows(data)


def read_csv(name):
    with codecs.open(name, 'r', encoding='utf-8') as f:
        table = []
        reader = csv.reader(f)
        for row in reader:
            table.append(row)
        return table


def print_histogram(datas, labels, axis=None, hist=True, kde=False, legend=True, save_dir='./result/histogram.png'):
    for data, label in zip(datas, labels):
        sns.distplot(data, hist=hist, kde=kde,
                     kde_kws={'linewidth': 3},
                     label=label)
    # Plot formatting
    if legend:
        plt.legend(prop={'size': 16}, loc=2)

    plt.title('')
    if axis:
        plt.axis(axis)
    plt.xlabel('KL Divergence')
    plt.ylabel('Number of models')
    plt.savefig(save_dir)


if __name__ == '__main__':
    import numpy as np
    # a = np.array([[3, 4, 5], [2, 3, 4]])
    # write_csv(data=a)
    # print_histogram([[1, 2, 3, 4]], labels=['a'], save_dir='../result/histogram.png')
    table = read_csv('../result/stocinf.csv')
    floats = []
    for row in table:
        for data in row:
            if data != '':
                floats.append(float(data))
    print(floats)
    print_histogram([floats], ["Stochastic"], axis=[74.5, 77, 0, 80], legend=False, save_dir='../result/histogram_1.png')

