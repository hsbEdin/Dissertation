from lxml import etree
import gzip
import json
import os
import click
import logging
#import ijson
#import ijson.backends.yajl2 as ijson
import gensim
# from gensim.models.phrases import Phrases, Phraser
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import pickle
import re
import annotate_text
import word2vec

def _log_passed_arguments(arguments):
    for key, value in arguments.items():
        if key is not 'self':
            logging.info('%s: %s', key, value)

def _get_files_in_folder(input_folder, ext=".xml.gz", start_file=None, end_file=None):
        currentDir = os.getcwd() if input_folder == '/' else input_folder
        start = os.path.join(currentDir, start_file) if start_file is not None else None
        end = os.path.join(currentDir, end_file) if end_file is not None else None
        files = []
        sf = False if start_file is not None else True
        for file in os.listdir(currentDir):
            if file.endswith(ext):
                files.append(os.path.join(currentDir, file))
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
        files = files[files.index(start) if start is not None else None : files.index(end) if end is not None else None ]
        return files

class PmParser(object):
    """
    Class providing various methods that are useful for parsing Pubmed articles

    """
    def __init__(self, logfile="parse.log"):
        logging.basicConfig(filename=logfile, level=logging.DEBUG, format='%(asctime)s %(message)s')

    # def _get_pubdate(article):
    #     # PubDate - lots of different variants to implement
    #     pass

    # def _get_sup_mesh(article):
    #     pass

    def _get_article_element(self, article):
        # Pubdate
        # SupMeshList
        geneSymbols = self._get_gene_symbol_list(article)
        keywords = self._get_keyword_list(article)
        meshHeadings = self._get_mesh_heading_list(article)
        pmid = self._get_pmid(article)
        articleTitle = self._get_title(article)
        abstractText = self._get_abstract(article)
        return geneSymbols, keywords, meshHeadings, pmid, articleTitle, abstractText

    @staticmethod
    def _get_pmid(article):
        pmid = article.xpath('.//PMID/text()')
        return pmid[0] if pmid is not None and len(pmid)>0 else None

    @staticmethod
    def _get_title(article):
        title = article.xpath('.//ArticleTitle/text()')
        return title[0] if title is not None and len(title)>0 else None

    @staticmethod
    def _get_abstract(article):
        abstract = article.xpath('.//AbstractText/text()')
        return abstract[0] if abstract is not None and len(abstract)>0 else None

    def _get_gene_symbol_list(self, article):
        parsedGeneSymbolList = []
        GeneSymbolList = article.xpath('.//GeneSymbolList')
        if self._xcheck(GeneSymbolList):
            genesymbols = GeneSymbolList[0].findall("GeneSymbol")
            for gene in genesymbols:
                    parsedGeneSymbolList.append({"term":gene.text})
        return parsedGeneSymbolList

    def _get_keyword_list(self, article):
        parsedKeywordList = []
        keywordList = article.xpath('.//KeywordList')
        if self._xcheck(keywordList):
            keywords = keywordList[0].findall("Keyword")
            for keyword in keywords:
                    parsedKeywordList.append({
                        "majorTopicYN":keyword.get('MajorTopicYN') == "Y",
                        "term":keyword.text})
        return parsedKeywordList

    def _get_mesh_heading_list(self, article):
        parsedMeshList = []
        meshHeadingList = article.xpath('.//MeshHeadingList')
        if self._xcheck(meshHeadingList):
            meshHeadings = meshHeadingList[0].findall('MeshHeading')
            for meshHeading in meshHeadings:
                descriptor = meshHeading.xpath('.//DescriptorName')
                qualifier = meshHeading.xpath('.//QualifierName')
                if self._xcheck(descriptor):
                    parsedMeshList.append({
                        "type":"Descriptor",
                        "majorTopicYN":descriptor[0].get('MajorTopicYN') == "Y",
                        "term":descriptor[0].text})
                elif self._xcheck(qualifier):
                    parsedMeshList.append({
                        "type":"Qualifier",
                        "majorTopicYN":qualifier[0].get('MajorTopicYN') == "Y",
                        "term":qualifier[0].text})
        return parsedMeshList

    @staticmethod
    def _xcheck(term):
        return term is not None and len(term)>0

    @staticmethod
    def _check_conditions(pmid, title, abstract, meshes, keywords):
        return pmid is not None and title is not None and abstract is not None and (len(meshes)>0 or len(keywords)>0)

    def fast_iter(self, context, func, *args, **kwargs):
        """
        http://lxml.de/parsing.html#modifying-the-tree
        Based on Liza Daly's fast_iter
        http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
        See also http://effbot.org/zone/element-iterparse.htm
        """
        parsed_articles = []
        total_articles = 0
        for event, elem in context:
            total_articles += 1
            geneSymbols, keywords, meshHeadings, pmid, articleTitle, abstractText = func(elem, *args, **kwargs)
            if self._check_conditions(pmid, articleTitle, abstractText, meshHeadings, keywords):
                parsed_articles.append({
                            "PMID": pmid,
                            "articleTitle":articleTitle,
                            "abstractText": abstractText,
                            "meshHeadings": meshHeadings,
                            "keywords": keywords,
                            "geneSymbols": geneSymbols
                        })
            # It's safe to call clear() here because no descendants will be
            # accessed
            elem.clear()
            # Also eliminate now-empty references from the root node to elem
            for ancestor in elem.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]
        del context
        logging.info('total articles parsed: %s', total_articles)
        return parsed_articles

    def _parse(self, file):
        try:
            with gzip.GzipFile(file, 'r') as fin:
                logging.info('etree iterparsing start')
                context = etree.iterparse(fin, tag='PubmedArticle')
                parsed_file = self.fast_iter(context, self._get_article_element)
                logging.info('etree iterparsing end')
                logging.info('suitable articles received: %s', str(len(parsed_file)))
                return parsed_file
        except Exception as e:
            logging.warning('Something went wrong: %s', e)

    def _save_as_json(self, articles, counter, output_folder, name):
        articlesJson = json.dumps(articles, separators=(',',':'))
        jsonfilename = os.path.join(output_folder, "{}{}.json.gz".format(name, counter))
        logging.debug('saved as %s',jsonfilename)
        json_bytes = articlesJson.encode('utf-8')
        try:
            with gzip.GzipFile(jsonfilename, 'wb+') as fout:
                fout.write(json_bytes)
                del json_bytes
                del articlesJson
                articles.clear()
        except Exception as e:
            logging.warning('Writing error: %s', e)

    def parse_file(self, input_file, output_folder, name):
        """
        Parses a single file and stores the output as a json.gz compressed file

        Parameters:
        input_file (string): path to input file
        output_folder (string): path to folder for storing output

        Returns:
        Nothing

        """
        _log_passed_arguments(locals())
        self._save_as_json(self._parse(input_file), 0, output_folder, name)

    def parse_folder(self, output_folder, ratio, start_file, end_file, name_count, ext, name, input_folder):
        """
        Parses a all xml.gz files in specified folder and stores the output as a json.gz compressed file

        Parameters:
        input_folder (string): path to folder containing input files
        ratio (int): ratio of input files to output files, if 0 creates 1 output file from all input files
        output_folder (string): path to folder for storing output
        start_file (string): file at which to start parsing
        end_file (string): file at which to end parsing
        name_count (int): output file name counter initial value
        ext (string): file extension to look for in folder
        name (string): file name template

        Returns:
        Nothing

        """
        _log_passed_arguments(locals())
        logging.warning('Starting folder parse ->')
        logging.info('Collecting files in folder')
        all_files = _get_files_in_folder(input_folder, ext, start_file, end_file)
        logging.debug('Files collected #%s', str(len(all_files)))
        correctArticleList = []
        counter = 0
        file_count = name_count
        with click.progressbar(all_files) as files:
            for file in files:
                logging.debug('Parsing file: %s', file)
                correctArticleList.extend(self._parse(file))
                logging.debug('File parsed')
                counter += 1
                if ratio != 0 and counter % ratio == 0 or counter == len(all_files):
                    self._save_as_json(correctArticleList, file_count, output_folder, name)
                    correctArticleList.clear()
                    file_count += 1
            if ratio == 0:
                logging.debug('Saving to json.gz')
                self._save_as_json(correctArticleList, file_count, output_folder, name)
                logging.debug('Saved.')
            logging.debug('Parsing end')
            logging.debug('___________')

    def load(self, file_path):
        """
        Reads json.gz file and loads the json object into memory

        Parameters:
        input_file (string): path to input file

        Returns:
        dict: json object loaded from memory

        """
        _log_passed_arguments(locals())
        with gzip.GzipFile(file_path, 'rb') as fin:
            data = json.loads(fin.read().decode('utf-8'))
            # print(data)
            return data

    def load_folder(self, file_path):
        """
        Reads json.gz files from a folder and loads their json objects into memory

        Parameters:
        input_file (string): path to input file

        Returns:
        dict: json object loaded from memory

        """
        _log_passed_arguments(locals())
        with gzip.GzipFile(file_path, 'rb') as fin:
            data = json.loads(fin.read().decode('utf-8'))
            return data

    def _check_keyword(self, keyword, meshHeadings, keywords):
        ukeywords = [k.lower() for k in keyword.split(',')]
        meshes = [mesh['term'].lower() for mesh in meshHeadings if mesh['term'] is not None]
        pkeywords = [keyw['term'].lower() for keyw in keywords if keyw['term'] is not None]
        for key in ukeywords:
            if key in meshes:
                return True
            elif key in pkeywords: #elif -> if ?
                return True
        return False

    #def _parse_json_to_text(self, input_file, keyword, output_handle):
    #    for item in ijson.items(gzip.GzipFile(input_file), 'item'):
    #        if keyword is not None:
    #            if self._check_keyword(keyword, item['meshHeadings'], item['keywords']):
    #                output_handle.write(item['articleTitle'].encode())
    #                output_handle.write(item['abstractText'].encode())
    #                output_handle.write('\n'.encode())
    #        else:
    #            output_handle.write(item['articleTitle'].encode())
    #            output_handle.write(item['abstractText'].encode())
    #            output_handle.write('\n'.encode())


    def parse_folder_to_text(self, output_folder, ratio, keyword, start_file, end_file, name_count, ext, name, input_folder):
        """
        Parses all xml.gz files in specified folder and stores the output as a json.gz compressed file

        Parameters:
        input_folder (string): path to folder containing input files
        ratio (int): ratio of input files to output files, if 0 creates 1 output file from all input files
        output_folder (string): path to folder for storing output
        keyword (string): word phrase to filter articles by, if none given all articles converted to text
        start_file (string): file at which to start parsing
        end_file (string): file at which to end parsing
        name_count (int): output file name counter initial value
        ext (string): file extension to look for in folder
        name (string): file name template

        Returns:
        Nothing

        """
        _log_passed_arguments(locals())
        logging.warning('Starting folder to text parse ->')
        logging.info('Collecting files in folder')
        all_files = _get_files_in_folder(input_folder, ext, start_file, end_file)
        logging.debug('Files collected #%s', str(len(all_files)))
        counter = 0
        file_count = name_count
        jsonfilename = os.path.join(output_folder, "{}{}.txt.gz".format(name, file_count))
        fout = gzip.GzipFile(jsonfilename, 'wb+')
        logging.debug('Text written to file %s', jsonfilename)
        with click.progressbar(all_files) as files:
            for file in files:
                logging.debug('Parsing file: %s', file)
                self._parse_json_to_text(file, keyword, fout)
                logging.debug('File parsed')
                counter += 1
                if ratio != 0 and counter % ratio == 0: #or counter == len(all_files)
                    fout.close()
                    file_count += 1
                    jsonfilename = os.path.join(output_folder, "{}{}.txt.gz".format(name, file_count))
                    fout = gzip.GzipFile(jsonfilename, 'wb+')
                    logging.debug('[Change] Text written to file %s', jsonfilename)
            fout.close()
            logging.debug('Parsing end')
            logging.debug('___________')
    def merge_txt_files(self, input_folder, output_file, start_file, end_file, ext):
        _log_passed_arguments(locals())
        logging.warning('Starting folder to text parse ->')
        logging.info('Collecting files in folder')
        all_files = _get_files_in_folder(input_folder, ext, start_file, end_file)
        logging.debug('Files collected #%s', str(len(all_files)))
        with click.progressbar(all_files) as files:
            with gzip.GzipFile(output_file, 'wb+') as outfile:
                for fname in files:
                    logging.debug('Merging file: %s', fname)
                    with gzip.GzipFile(fname, 'rb') as infile:
                        for line in infile:
                            outfile.write(line)

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
        _log_passed_arguments(locals())
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
        _log_passed_arguments(locals())
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

def get_mwe_tokenizer(tokens_file):
    with open(tokens_file, 'rb') as f:
        data = pickle.load(f)
        tokenizer = MWETokenizer(data)
        return tokenizer

def tokenize_abstract(abstract, tokenizer):
    tokens = [w.lower() for w in tokenizer.tokenize(word_tokenize(abstract))]
    return tokens

def load_phrase_vectors(model_file):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    return model

#abstractEX = 'Priorities for Advancing Research on Youth with Autism Spectrum Disorder and Co-occurring Anxiety.Research on anxiety disorders in youth with autism spectrum disorder (ASD) has burgeoned in the past two decades. Yet, critical gaps exist with respect to measuring and treating anxiety in this population. This study used the nominal group technique to identify the most important research priorities on co-occurring anxiety in ASD. An international group of researchers and clinicians with experience in ASD and anxiety participated in the process. Topics ranked as most important focused on understanding how ASD symptoms affect treatment response, implementing treatments in real world settings, developing methods to disentangle overlapping symptoms between anxiety and ASD, and developing objective measures to assess anxiety. Collectively, these priorities can lead to collaborative studies to accelerate research in the field.'
#abstractEX = 'Priorities for Advancing Research on Youth with Autism Spectrum Disorder and Co-occurring Anxiety.'
#abstractEX = 'Impaired lipid metabolism markers to assess the risk of neuroinflammation in autism spectrum disorder. Autism spectrum disorder (ASD) is a multifactorial disorder caused by an interaction between environmental risk factors and a genetic background. It is characterized by impairment in communication, social interaction, repetitive behavior, and sensory processing. The etiology of ASD is still not fully understood, and the role of neuroinflammation in autism behaviors needs to be further investigated. The aim of the present study was to test the possible association between prostaglandin E2 (PGE2), cyclooxygenase-2 (COX-2), microsomal prostaglandin E synthase-1 (mPGES-1), prostaglandin PGE2 EP2 receptors and nuclear kappa B (NF-κB) and the severity of cognitive disorders, social impairment, and sensory dysfunction. PGE2, COX-2, mPGES-1, PGE2-EP2 receptors and NF-κB as biochemical parameters related to neuroinflammation were determined in the plasma of 47 Saudi male patients with ASD, categorized as mild to moderate and severe as indicated by the Childhood Autism Rating Scale (CARS) or the Social Responsiveness Scale (SRS) or the Short Sensory Profile (SSP) and compared to 46 neurotypical controls. The data indicated that ASD patients have remarkably higher levels of the measured parameters compared to neurotypical controls, except for EP2 receptors that showed an opposite trend. While the measured parameter did not correlate with the severity of social and cognitive dysfunction, PGE2, COX-2, and mPGES-1 were remarkably associated with the dysfunction in sensory processing. NF-κB was significantly increased in relation to age. Based on the discussed data, the positive correlation between PGE2, COX-2, and mPGES-1 confirm the role of PGE2 pathway and neuroinflammation in the etiology of ASD, and the possibility of using PGE2, COX-2 and mPGES-1 as biomarkers of autism severity. NF-κB as inflammatory inducer showed an elevated level in plasma of ASD individuals. Receiver operating characteristic analysis together with predictiveness diagrams proved that the measured parameters could be used as predictive biomarkers of biochemical correlates to ASD.'
#def evaluate(abstract, phe, tokens_file='mwe_tokens.pickle', model_file='vectors-phrase.bin'):
def evaluate(abstract, phe, tokens_file='mwe_tokens.pickle', model_file='test_word2vec.txt'):
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
