import pickle
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
import gensim
import csv
import annotate_text
import json

def get_hp_phenotypes():
    asd_terms = []
    hp_terms = []
    
    with open('ASD_phenotype.txt', 'r') as f:
        for line in f:
            asd_terms.append(line)
    with open('All_Phenotype.txt', 'r') as af:
        for line in af:
            if line not in asd_terms:
                hp_terms.append(line)
    with open('HP_phenotype.txt', 'w+') as hf:
        for term in hp_terms:
            hf.write(term)

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

def parse_onto_labels(input1="ASDPTO.csv", input2="HP.csv"):
    phe = {}
    with open(input1,"r") as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            label = line[1]
            if label == 'Preferred Label':
                continue
            label = label.lower()
            label = label.replace('-', '_')
            label = label.replace(' ', '_')
            phe[label] = '_asdpto'
        
    with open(input2,"r") as csvfile:
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
    with open('onto_tokens.pickle', 'wb+') as o:
            pickle.dump(phe, o)

def load_phenotypes(phenotypes_pickle):
    with open(phenotypes_pickle, 'rb') as f:
        data = pickle.load(f)
        return data
        
def get_mwe_tokenizer(tokens_file):
    with open(tokens_file, 'rb') as f:
        data = pickle.load(f)
        tokenizer = MWETokenizer(data)
        return tokenizer

def tokenize_abstract(abstract, tokenizer):
    tokens = [w.lower() for w in tokenizer.tokenize(word_tokenize(abstract))]
    return tokens

def load_phrase_vectors(model_file):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
    return model

abstractEX = 'Priorities for Advancing Research on Youth with Autism Spectrum Disorder and Co-occurring Anxiety.Research on anxiety disorders in youth with autism spectrum disorder (ASD) has burgeoned in the past two decades. Yet, critical gaps exist with respect to measuring and treating anxiety in this population. This study used the nominal group technique to identify the most important research priorities on co-occurring anxiety in ASD. An international group of researchers and clinicians with experience in ASD and anxiety participated in the process. Topics ranked as most important focused on understanding how ASD symptoms affect treatment response, implementing treatments in real world settings, developing methods to disentangle overlapping symptoms between anxiety and ASD, and developing objective measures to assess anxiety. Collectively, these priorities can lead to collaborative studies to accelerate research in the field.'
def evaluate_abstract(abstract=abstractEX, tokens_file='/afs/inf.ed.ac.uk/group/project/biomednlp/msc2019/maciej/data/mwe_tokens.pickle', model_file='/afs/inf.ed.ac.uk/group/project/biomednlp/msc2019/Shibo/word2vec/examples/vectors-phrase.bin'):
    tokenizer = get_mwe_tokenizer(tokens_file)
    tokens = tokenize_abstract(abstract, tokenizer)
    model = load_phrase_vectors(model_file)
    for token in tokens:
        try:
            similars = model.most_similar(token)
            for term, score in similars:
                if score >= 0.90 and ('asdpto' in term or 'hp' in term):
                    print(token, '-', term)
        except Exception as e:
            print('Error', e)

def evaluate(abstract, phe, tokens_file='mwe_tokens.pickle', model_file='vectors-phrase.bin'):
    print("The NCBO anntator: ")
    annotate_text.print_annotations(abstract)
    print("\n")
    print("The word2vec model: ")
    abstract = abstract.replace('.', ' ')
    abstract = abstract.replace('-', ' ')
    tokenizer = get_mwe_tokenizer(tokens_file)
    tokens = tokenize_abstract(abstract, tokenizer)
    model = load_phrase_vectors(model_file)
    phenotype = {}
    for token in tokens:
        try:
            if token in phe.keys():#Find the abstract level word that is the same as the phenotype
                if token not in phenotype.keys():
                    phenotype[token] = str(token) + str(phe[token])

            similars = model.most_similar(token)
            for term, score in similars:
                if score >= 0.90:
                    if term[-2:] == 'hp' and (re.search('_hp',term).span()[1] == len(term)):
                        #print(token, '-', term)
                        if token in phenotype.keys():
                            if term in phenotype[token]:
                                continue
                            else:
                                phenotype[token] += ', ' + term
                        else:
                            phenotype[token] = term
                    elif term[-6:] == 'asdpto' and (re.search('_asdpto',term).span()[1] == len(term)):
                        #print(token, '-', term)
                        if token in phenotype.keys():
                            continue
                        else:
                            phenotype[token] = term
                #if score >= 0.90 and ('asdpto' in term or 'hp' in term):
        except Exception as e:
            #continue
            print('Error', e)
    for key, value in phenotype.items():
        print(key, '-', value)
    print("\n")

