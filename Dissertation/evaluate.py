import pm_parser
import csv

phe = {}
with open("ASDPTO.csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        label = line[1]
        if label == 'Preferred Label':
            continue
        label = label.lower()
        label = label.replace('-', '_')
        label = label.replace(' ', '_')
        phe[label] = '_asdpto'


with open("HP.csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        label = line[1]
        if label == 'Preferred Label':
            continue
        elif label[0:3] == 'HP_':
            continue
        label = label.lower()
        label = label.replace('-', '_')
        label = label.replace(' ', '_')
        if label in phe.keys():
            phe[label] += '_hp'
        else:
            phe[label] = '_hp'

filename = 'Autism_Pure_Testing.txt'

with open(filename, 'r') as file_to_read:
    while True:
        text_to_annotate = file_to_read.readline()
        pm_parser.evaluate(text_to_annotate, phe)
