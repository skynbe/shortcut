import sys
import csv
import codecs

def write_csv(name='output.csv', header=None, data=None):
    '''
    header: name of column. (ex) ["Name", "Accuracy", "Loss"]
    data: 2-dim array
    '''
    # if sys.version_info.major==2:
    with codecs.open(name, 'w', encoding='utf-8') as f:
        wr = csv.writer(f)
        if header:
            wr.writerow(header)
        wr.writerows(data)


if __name__ == '__main__':
    import numpy as np
    a = np.array([[3, 4, 5], [2, 3, 4]])
    write_csv(data=a)

