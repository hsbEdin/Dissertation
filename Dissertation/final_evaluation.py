import re
import dfrg_w2v
import json
import pickle
# import click
from tqdm import tqdm
from datetime import datetime
import utils
import csv

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def check_hp_term(term):
    return term[-2:] == 'hp' and (re.search('_hp',term).span()[1] == len(term))

def check_asdpto_term(term):
    return term[-6:] == 'asdpto' and (re.search('_asdpto',term).span()[1] == len(term))

def evaluate_folder(folder, output_file, annotated_testset_file="annotated_testset.json", phenotypes_pickle="onto_tokens.pickle", mwe_tokens_pickle="mwe_tokens.pickle", ext=".model", start_file=None, end_file=None):
    models = utils._get_files_in_folder(folder, start_file=start_file, end_file=end_file, ext=ext)
    print('Number of models to evaluate:', len(models))
    with open(output_file, mode='a') as out:
        fieldnames = ['model', 'eval_time', 'num_of_abstracts', 'f1', 'precision', 'recall']
        results_writer = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(fieldnames)
        for model in tqdm(models): 
            eval_time, num_of_abstracts, precision, recall, f1 = evaluation(annotated_testset_file=annotated_testset_file, phenotypes_pickle=phenotypes_pickle, mwe_tokens_pickle=mwe_tokens_pickle, model_file=model)
            results_writer.writerow([model, eval_time, num_of_abstracts, f1, precision, recall])
        print('****finished****')

def evaluation(annotated_testset_file="annotated_testset.json", phenotypes_pickle="onto_tokens.pickle", mwe_tokens_pickle="mwe_tokens.pickle", modelb=None, model_file="experiments/1_dataset_selection/autism_model.bin", verbose=False, summary=False, sim_score=0.90):
    start = datetime.now()
    tokenizer = dfrg_w2v.get_mwe_tokenizer(mwe_tokens_pickle)
    model = dfrg_w2v.load_phrase_vectors(model_file) if modelb is None else modelb
    phe = dfrg_w2v.load_phenotypes(phenotypes_pickle)
    gmatches = 0
    gmismatches = 0
    grelevant = 0
    with open(annotated_testset_file, 'r') as json_file:
        data = json.load(json_file)
        if summary:
            print('\nModel:', model_file)
            print('Number of abstracts:', len(data))
        for abstractObj in tqdm(data): 
            matches, mismatches, relevant = evaluate_abstract(abstractObj, tokenizer, model, phe, verbose=verbose, sim_score=sim_score)
            gmatches += matches
            gmismatches += mismatches
            grelevant += relevant
        precision = (gmatches)/(gmatches + gmismatches)
        recall = (gmatches)/grelevant
        f1 = 2*precision*recall/(precision+recall)
        end = datetime.now() - start
        if summary:
            print('________')
            print('STATS:')
            print('Time taken:', end)
            print('Number of abstracts:', len(data))
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1:', f1)
            print('')
        return (end, len(data), precision, recall, f1)

def evaluate_abstract(abstractObj, tokenizer, model, phe, verbose=False, sim_score=0.90):
    abstract = abstractObj["abstract"].replace('.', ' ').replace('-', ' ').replace(',',' ').replace('(',' ').replace(')',' ').replace(':',' ')
    tokens = dfrg_w2v.tokenize_abstract(abstract, tokenizer)
    onto_annos = list(set([anno['onto_lvl'].lower() for anno in abstractObj['annotations']]))
    phenotype_annos = list(set([anno['abstract_lvl'].lower() for anno in abstractObj['annotations']])) 
    found_phenotypes = []
    match_count = 0 
    mismatch_count = 0
    relevant = len(onto_annos)
    errors = []
    if verbose:
        print('\nAbstract:', tokens)
        print(abstract)
        print('Annos:')
        for anno in onto_annos:
            print(bcolors.WARNING + anno + bcolors.ENDC)
    for token in tokens:
        try:
            similars = model.most_similar(token)
            for term, score in similars:
                # if score >= 0.90 and (check_hp_term(term) or check_asdpto_term(term)):
                if score >= sim_score and (check_hp_term(term) or check_asdpto_term(term)):
                    found_phenotypes.append(term)
        except Exception as e: 
            errors.append(str(e).split('\'')[1])
            # pass
    for found_phenotype in list(set(found_phenotypes)):
        match = [onto for onto in onto_annos if found_phenotype in onto]
        if len(match) > 0:
            match_count += 1
            if verbose:
                info = "FP:{} Onto:{}".format(found_phenotype, ', '.join(match))
                print(bcolors.OKGREEN+info+bcolors.ENDC)
        else:
            mismatch_count += 1
            if verbose:
                info = "FP:{}".format(found_phenotype)
                print(bcolors.FAIL+info+bcolors.ENDC)
    if verbose:
        print('Matches: ', match_count, ' Mismatches: ', mismatch_count)
        print('errors')
        for er in list(set(errors)):
            print(er)
        print("")
    return (match_count, mismatch_count, relevant)

def evaluate_gn_abstract(abstractObj, verbose=False):
    ncbo_onto_annos = list(set([anno['onto_lvl'].lower() for anno in abstractObj['ncbo_annotations']]))
    ncbo_phenotype_annos = list(set([anno['abstract_lvl'].lower() for anno in abstractObj['ncbo_annotations']]))
    gold_onto_annos = list(set([anno['onto_lvl'].lower() for anno in abstractObj['gold_annotations']]))
    gold_phenotype_annos = list(set([anno['abstract_lvl'].lower() for anno in abstractObj['gold_annotations']])) 
    found_phenotypes = []
    match_count = 0 
    mismatch_count = 0
    relevant = len(gold_onto_annos)

    if verbose:
        print('Abstract:', abstractObj['abstract'])
        print('Annos:', abstractObj['gold_annotations'])
    for found_phenotype in ncbo_onto_annos:
        if check_hp_term(found_phenotype):
            print('ok')
            match = [onto for onto in gold_onto_annos if found_phenotype in onto]
            if len(match) > 0:
                match_count += 1
                if verbose:
                    info = "FP:{} Onto:{}".format(found_phenotype, ', '.join(match))
                    print(bcolors.OKGREEN+info+bcolors.ENDC)
            else:
                mismatch_count += 1
                if verbose:
                    info = "FP:{}".format(found_phenotype)
                    print(bcolors.FAIL+info+bcolors.ENDC)
        else:
            print('NOT OKc')
    if verbose:
        print('Matches: ', match_count, ' Mismatches: ', mismatch_count)
        print("")
    return (match_count, mismatch_count, relevant)

def ncbo_evaluation(gold_ncbo_annotation_file="annotated_GOLD_NCBO_testset.json", outfile='results_ncbo_gold.csv', verbose=False):
    start = datetime.now()
    gmatches = 0
    gmismatches = 0
    grelevant = 0
    with open(gold_ncbo_annotation_file, 'r') as gold_json:
        data = json.load(gold_json)
        num_of_abstracts = len(data)
        for abstractObj in tqdm(data): 
            matches, mismatches, relevant = evaluate_gn_abstract(abstractObj, verbose=verbose)
            gmatches += matches
            gmismatches += mismatches
            grelevant += relevant
        precision = (gmatches)/(gmatches + gmismatches)
        recall = (gmatches)/grelevant
        f1 = 2*precision*recall/(precision+recall)
        end = datetime.now() - start
        print('________')
        print('STATS:')
        print('Time taken:', end)
        print('Number of abstracts:', len(data))
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1:', f1)
        print('')
        with open(outfile, mode='a') as out:
            fieldnames = ['model', 'eval_time', 'num_of_abstracts', 'f1', 'precision', 'recall']
            results_writer = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(fieldnames)
            results_writer.writerow(['ncbo_annotator', end, num_of_abstracts, f1, precision, recall])
        return (end, len(data), precision, recall, f1)

def sim_score_eval(output_file='sim_score_ncbo_results2.csv', annotated_testset_file="annotated_testset.json", phenotypes_pickle="onto_tokens.pickle", mwe_tokens_pickle="mwe_tokens.pickle", modelb=None, model_file="experiments/1_dataset_selection/autism_model.bin", verbose=False, summary=False):
    # sim_scores = [0.84,0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96]
    sim_scores = [0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96]

    print('Number of scores to evaluate:', len(sim_scores))
    with open(output_file, mode='a') as out:
        fieldnames = ['sim_score', 'eval_time', 'num_of_abstracts', 'f1', 'precision', 'recall']
        results_writer = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(fieldnames)
        for sim_score in tqdm(sim_scores): 
            eval_time, num_of_abstracts, precision, recall, f1 = evaluation(annotated_testset_file=annotated_testset_file, phenotypes_pickle=phenotypes_pickle, mwe_tokens_pickle=mwe_tokens_pickle, model_file=model_file, sim_score=sim_score)
            results_writer.writerow([sim_score, eval_time, num_of_abstracts, f1, precision, recall])
        print('****finished****')