import logging
import os
import gensim
import gzip
import string
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
from utils import _log_passed_arguments, _get_files_in_folder

class W2V(object):
    """
    Class wrapping the gensim word2vec method

    """
    def __init__(self, logfile="parse.log"):
        logging.basicConfig(filename=logfile, level=logging.DEBUG, format='%(asctime)s %(message)s')

    def _save_model(self, model, output_folder, output_file):
        currentDir = os.getcwd() if output_folder == '/' else output_folder
        model_file = os.path.join(currentDir, output_file)
        if model_file.endswith('.bin'):
            model.wv.save_word2vec_format(model_file, binary=True)
        else:
            model.save(model_file)
        logging.debug('Model saved as: %s', model_file)

    def intersect(self, pretrained_model, input_folder, output_folder, output_file, preprop, size, window, min_count, workers, log_file, atype, start_file, end_file, ext, input_file, gram):
        """ 
        Uses binary pretrained word2vec model to train word2vec word-embedding model and saves the trained model 
    
        Parameters: 
        pretrained_model (string): path to pretrained model file
        input_folder (string): path to folder containing input files
        output_folder (string): path to folder for storing output
        output_file (string): model file name
        preprop (int): number indicating preprocessing level
        size (int): resulting vector size
        window (int): window size
        min_count (int): minimal number of occurences required for a word to be a vocab constituent
        workers (int): number of workers used during training
        atype (int): training algorithm - 0 CBOW, 1 Skipgram
        start_file (string): file at which to start parsing
        end_file (string): file at which to end parsing
        ext (string): file extension to look for in folder
        input_file (string): single file name to be used
        gram (string): 'word' for simple word embedding, 'phrase' for more complex phrase embedding

    
        Returns: 
        Nothing 
    
        """
        # sentences = [["bad","robots"],["good","human"],['yes', 'this', 'is', 'the', 'word2vec', 'model']]
 
        # size option needs to be set to 300 to be the same as Google's pre-trained model
        _log_passed_arguments(logging, locals())
        sentences = MySentences(input_folder, preprop, start_file, end_file, ext, input_file, gram) # a memory-friendly iterator
     
        word2vec_model = gensim.models.Word2Vec(sample=1e-5, alpha=0.025, negative=25, iter=15, hs=0, size=size, window=window, min_count=min_count, workers=workers, sg=atype)
        logging.debug('new sentences loaded')

        word2vec_model.build_vocab(sentences) 
        logging.debug('Vocabulary built')      
        
        # assign the vectors to the vocabs that are in Google's pre-trained model and your sentences defined above.  
        # lockf needs to be set to 1.0 to allow continued training.
        word2vec_model.intersect_word2vec_format(pretrained_model, lockf=1.0, binary=True)
        logging.debug('Intersection complete')

        # continue training with you own data
        word2vec_model.train(sentences, total_examples=3, epochs = 5)
        logging.debug('Post intersection training complete')
        output_file = output_file if output_file is not None else "inter_{}_{}_{}.model".format(preprop, gram, window)
        self._save_model(word2vec_model, output_folder, output_file)

    def bin2text(self, input_file, output_file):
        """ 
        Converts binary pretrained word2vec model and saves it as a text file 
    
        Parameters: 
        input_file (string): binary model file name
        output_file (string): text model file name  
    
        Returns: 
        Nothing 
    
        """
        logging.debug('loading binary model: %s', input_file)
        model = gensim.models.KeyedVectors.load_word2vec_format('bio_nlp_vec/PubMed-shuffle-win-2.bin', binary=True)
        logging.debug('model loaded')
        logging.debug('saving to txt: %s', output_file)
        model.save_word2vec_format('bio_nlp_vec/PubMed-shuffle-win-2.txt', binary=False)
        logging.debug('model saved as txt file')

    def train(self, input_folder, output_folder, output_file, preprop, size=200, window=10, min_count=0, workers=8, atype=1, start_file=None, end_file=None, ext="txt.gz", input_file=None, gram='word'):
        """ 
        Trains word2vec word-embedding model and saves the trained model 
    
        Parameters: 
        input_folder (string): path to folder containing input files
        output_folder (string): path to folder for storing output
        output_file (string): model file name
        preprop (int): number indicating preprocessing level
        size (int): resulting vector size
        window (int): window size
        min_count (int): minimal number of occurences required for a word to be a vocab constituent
        workers (int): number of workers used during training
        atype (int): training algorithm - 0 CBOW, 1 Skipgram
        start_file (string): file at which to start parsing
        end_file (string): file at which to end parsing
        ext (string): file extension to look for in folder
        input_file (string): single file name to be used
        gram (string): 'word' for simple word embedding, 'phrase' for more complex phrase embedding
  
    
        Returns: 
        Nothing 
    
        """
        _log_passed_arguments(logging, locals())
        sentences = MySentences(input_folder, preprop, start_file, end_file, ext, input_file, gram) # a memory-friendly iterator
        model = gensim.models.Word2Vec(sample=1e-5, alpha=0.025, negative=25, iter=15, hs=0, size=size, window=window, min_count=min_count, workers=workers, sg=atype)
        model.build_vocab(sentences) 
        model.train(sentences, total_examples=3, epochs = 5)
        output_file = output_file if output_file is not None else "train{}_{}_{}.model".format(preprop, gram, window)
        self._save_model(model, output_folder, output_file)

class MySentences(object):
    """
    Class to be used by the gensim word2vec() method as an interator supplying sentences from files within a provided folder

    """
    def __init__(self, dirname, preprop, start_file, end_file, ext="txt.gz", input_file=None, gram='word'):
        self.dirname = dirname
        self.start_file = start_file
        self.end_file = end_file
        self.ext = ext
        self.preprop = preprop
        self.stop_words = set(stopwords.words('english'))
        self.files = _get_files_in_folder(self.dirname, self.ext, self.start_file, self.end_file) if input_file is None else [os.path.join(dirname, input_file)] 
        self.preprocessor = {0: self.preprop0, 1: self.preprop1, 2: self.preprop2, 3: self.preprop3, 4: self.preprop4, 5: self.preprop5, 6: self.preprop6, 7: self.preprop7}
        self.gram = gram
        self.stemmer = PorterStemmer()
        self.onto_terms = self.load_ontology_terms('/afs/inf.ed.ac.uk/group/project/biomednlp/msc2019/maciej/All_Phenotype.txt')
        self.tokenizer = MWETokenizer()
        for term in self.onto_terms:
            # print('++++',term.split())
            self.tokenizer.add_mwe(tuple(term.split()))
        # self.tokenizer.tokenize('Something about the west wing'.split())
        # Autism spectrum disorder - > Autism_spectrum_disorder
        # Autism spectrum disorder_HP - > Autism spectrum disorder_HP
        # asd 
        # autism_spectrum_disorder_hp
    def __iter__(self):
        for fname in self.files:
            logging.debug('Collecting file: %s', fname)
            file = gzip.open(os.path.join(self.dirname, fname), 'rt') if fname.endswith('.gz') else open(os.path.join(self.dirname, fname), 'r')
            text = file.read()
            logging.debug('File collected')
            sents = sent_tokenize(text)
            logging.debug('File sentence tokenized')
            if self.gram == 'phrase':
                phrases = gensim.models.phrases.Phrases([self.preprocessor[self.preprop](sent) for sent in sents], min_count=1, threshold=2)
                bigram = gensim.models.phrases.Phraser(phrases)
                for sent in sents:
                    
                    print(bigram[self.preprocessor[self.preprop](sent)])
                    yield bigram[self.preprocessor[self.preprop](sent)]
            else:
                for sent in sents:
                    yield self.preprocessor[self.preprop](sent)

    def load_ontology_terms(self, file):
        terms = []
        with open(file, 'r') as f:
            for line in f:
                terms.append(line.lower())
        return terms

    def preprop0(self, sent):
        return sent.split()
    
    def preprop1(self, sent):
        return gensim.utils.simple_preprocess(sent)
    
    def preprop2(self, sent):
        return [w for w in word_tokenize(sent) if not w in self.stop_words]

    def preprop3(self, sent):
        return [w.lower() for w in word_tokenize(sent) if not w in self.stop_words and w not in string.punctuation]
    
    def preprop4(self, sent):
        return [self.stemmer.stem(w.lower()) for w in word_tokenize(sent) if not w in self.stop_words]

    def preprop5(self, sent):
        return [w.lower() for w in self.tokenizer.tokenize(word_tokenize(sent))]

    def preprop6(self, sent):
        return [w.lower() for w in self.tokenizer.tokenize(word_tokenize(sent)) if w not in self.stop_words and w not in string.punctuation]

    def preprop7(self, sent):
        return [w.lower() for w in word_tokenize(sent) if w not in string.punctuation]