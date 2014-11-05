import numpy as np

files = {
    'preds_20_5x5_50_3x3_125_3x3_500.csv': [],
    'preds_20_9x9_60_5x5_180_3x3_500.csv': [],
    'preds_2conv_10000.csv': [],
    'preds_2conv_15000.csv': [],
    'preds_2conv_20000.csv': [],
    'preds_2conv_5000.csv': [],
    'preds_3conv_10000.csv': [],
    'preds_3conv_15000.csv': [],
    'preds_3conv_5000.csv': [],
    'preds_calvin_10000.csv': [],
    'preds_calvin_11000.csv': [],
    'preds_calvin_12000.csv': [],
    'preds_calvin_13000.csv': [],
    'preds_calvin_14000.csv': [],
    'preds_calvin_15000.csv': [],
    'preds_calvin_9000.csv': [],
    'preds_rotated.csv': []
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
        files[filename].append(int(line[1]))

f = open('common.csv', 'wb')
f.write('Id,Prediction\n')
for id in xrange(1, 20001):
    p = [0] * 10
    for filename in files:
        pred = files[filename][id-1]
        p[pred] += 1
    #print p
    if max(p) == len(files.keys()):
      f.write('%d,%d\n' % (id,np.argmax(p)))
      #f.write('../test_images/%d.png %d\n' % (id,np.argmax(p)))
    else:
      print p
f.close()
