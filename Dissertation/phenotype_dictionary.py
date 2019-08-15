import pickle
import csv
from nltk.tokenize import word_tokenize


phe = {}
with open("HP.csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        label = line[1]
        if label == 'Preferred Label':
            continue
        elif label[0:3] == 'HP_':
            continue

        if label:
            if '-' in label:
                label = label.replace('-',' ')
            phe[label] = []
            #print(label[0:2])
            if line[2]:
                for i in line[2].split('|'):
                    if '-' in i:
                        i = i.replace('-',' ')
                    phe[label].append(i)

        label = ''
with open("ASDPTO.csv","r") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        label = line[1]
        if label == 'Preferred Label':
            continue

        if label:
            if '-' in label:
                label = label.replace('-',' ')
            phe[label] = []

        label = ''

with open('phenotype_dictionary.pickle', 'wb') as handle:
    pickle.dump(phe, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('phenotype_dictionary.pickle', 'rb') as handle:
    new = pickle.load(handle)

print(new == phe)

def onto_2_mwe_tokens(input_file, output_file):
    with open(input_file, 'rb') as f:
        mwe_tokens_list = []
        data = pickle.load(f)
        for label, synonyms in data.items():
            mwe_tokens_list.append(tuple(w.lower() for w in word_tokenize(label)))
            for synonym in synonyms:
                mwe_tokens_list.append(tuple(w.lower() for w in word_tokenize(synonym)))
        with open(output_file, 'wb+') as o:
            pickle.dump(mwe_tokens_list, o)

onto_2_mwe_tokens('phenotype_dictionary.pickle', 'mwe_tokens.pickle')
        #phe[label] = syn.split('|')

#print(phe)

#with open('filename.pickle', 'wb') as handle:
#    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
