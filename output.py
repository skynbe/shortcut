import matplotlib
matplotlib.use('Agg')
import sys
import csv
import codecs
import matplotlib.pyplot as plt
import seaborn as sns


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


def print_histogram(datas, labels, hist=True, kde=True, save_dir='./result/histogram3.png'):
    for data, label in zip(datas, labels):
        sns.distplot(data, hist=hist, kde=kde,
                     kde_kws={'linewidth': 3},
                     label=label)
    # Plot formatting
    leg = plt.legend(prop={'size': 16}, title='Plot')
    leg._legend_box.align = "left"
    plt.title('Density Plot')
    plt.xlabel('KL divg')
    plt.ylabel('Density')
    plt.savefig(save_dir)


if __name__ == '__main__':
    import numpy as np
    # a = np.array([[3, 4, 5], [2, 3, 4]])
    # write_csv(data=a)
    print_histogram([[1, 2, 3, 4]], labels=['a'], save_dir='../result/histogram.png')

