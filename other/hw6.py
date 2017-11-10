import pandas as pd


ex1 = pd.read_csv(filepath_or_buffer='predictions.csv', header=0)

true_pos = 0
tn = 0
false_pos = 0
fn = 0

for index, row in ex1.iterrows():
    if row.B == 1 and row.Actual == 1:
        true_pos += 1
    if row.B == 0 and row.Actual == 1:
        fn += 1
    if row.B == 1 and row.Actual == 0:
        false_pos += 1
    if row.B == 0 and row.Actual == 0:
        tn += 1

print(true_pos)
print(tn)
print(false_pos)
print(fn)

def accuracy(tp, tn):
    return (tp + tn)/ 200

def precision(tp, pos):
    return tp / pos

def recall(tp, pos):
    return tp / pos

def fmeasure(tp, fn, fp):
    return 2 / ( ( 1 / precision(tp, tp+fp)) + ( 1 / recall(tp, tp+fn)))

print(accuracy(true_pos, tn))
print(precision(true_pos, true_pos + false_pos))
print(recall(true_pos, true_pos + fn))
print(fmeasure(true_pos, fn, false_pos))