import numpy as np

files = {
        'Downloads/ensemble.csv': [],
        'preds_20_9x9_60_5x5_180_3x3_500.csv': [],
        'Downloads/preds_nn_2304x280x10_100epochs.csv': [],
        'preds_calvin_10000.csv': [],
        }

for filename in files:
    f = open(filename)
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.split(',')
        files[filename].append((line[0], int(line[1])))

f = open('ensemble.csv', 'wb')
f.write('Id,Prediction\n')
for id in xrange(1, 20001):
    p = [0] * 10
    for filename in files:
        pred = files[filename][id-1][1]
        p[pred] += 1
    f.write('%d,%d\n' % (id, np.argmax(p)))
f.close()
