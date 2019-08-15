import csv
import pickle

hp = {}
with open("HP.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        if line[0] == "Class ID":
            continue
        #if line[0] != line[1]:
        hp[line[0]] = line[1]

test = {}
with open("pmc2hp.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        for i in line:
            if i == line[0]:
                test[i] = {}
                continue
            test[line[0]][i] = ''

for key in test.keys():
    for key1 in test[key].keys():
        test[key][key1] = hp[key1]
        #if test[key][key1] in hp.keys():
        #    test[key][key1] = hp[key1]

for key, value in test.items():
    print(key, ' ', value)

with open('terms.pickle', 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('terms.pickle', 'rb') as handle:
    new = pickle.load(handle)

print(new == test)

file = open('terms.txt', 'a')
file.write(test)
file.close()
