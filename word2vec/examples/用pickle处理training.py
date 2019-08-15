import pm_parser
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
import pickle


def get_mwe_tokenizer(tokens_file):
    with open(tokens_file, 'rb') as f:
        data = pickle.load(f)
        tokenizer = MWETokenizer(data)
        return tokenizer

def tokenize_abstract(abstract, tokenizer):
    tokens = [w.lower() for w in tokenizer.tokenize(word_tokenize(abstract))]
    return tokens

filename = 'Training.txt'
sym = [',', '.', ';', ':', '?', ')', '%', '*']
t = 0
tokenizer = get_mwe_tokenizer('mwe_tokens.pickle')
with open(filename, 'r') as file_to_read:
    while True:
        text_to_annotate = file_to_read.readline()
        abstract = ''
        #pm_parser.evaluate(text_to_annotate)
        tokens = tokenize_abstract(text_to_annotate, tokenizer)
        for token in tokens:
            if token in sym:
                abstract = abstract[:-1]
                abstract += token + ' '
            elif token == '(':
                abstract += token
            else:
                abstract += token + ' '
        file = open('New_Training.txt', 'a')
        file.write(abstract)
        file.close()
