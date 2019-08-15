import urllib.request, urllib.error, urllib.parse
import json
import os
from pprint import pprint

REST_URL = "http://data.bioontology.org"
API_KEY = "dd837192-d0d9-493c-bf6d-6e20845fa8f8"

def get_json(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return json.loads(opener.open(url).read())

def print_annotations(text_to_annotate, get_class=True):
    annotations = get_json(REST_URL + "/annotator?text=" + urllib.parse.quote(text_to_annotate) + '&ontologies=HP,ASDPTO')
    phenotype = {}
    for result in annotations:
        class_details = result["annotatedClass"]
        if get_class:
            try:
                class_details = get_json(result["annotatedClass"]["links"]["self"])
            except urllib.error.HTTPError:
                print(f"Error retrieving {result['annotatedClass']['@id']}")
                continue
        #id = ['http://data.bioontology.org/ontologies/ASDPTO','http://data.bioontology.org/ontologies/HP']
        #if class_details["links"]["ontology"] in id:

        #print("Class details")
        #print("\tid: " + class_details["@id"])
        #print("\tprefLabel: " + class_details["prefLabel"])
        #print("\tontology: " + class_details["links"]["ontology"])
        #print("Annotation details")
        if 'ASDPTO' in class_details["links"]["ontology"]:
            onto = 'ASDPTO'
        else:
            onto = 'HP'
        for annotation in result["annotations"]:
            #print("\tfrom: " + str(annotation["from"]))
            #print("\tto: " + str(annotation["to"]))
            x = int(annotation["from"]) - 1
            y = int(annotation["to"])
            word = text_to_annotate[x:y]
            label = class_details["prefLabel"]
            label = label.replace(' ','_') + '_' + onto
            word = word.lower()
            if word in phenotype.keys():
                if label in phenotype[word]:
                    continue
                else:
                    phenotype[word] += ', ' + label
            else:
                phenotype[word] = label
            #print(word, '-', label+'_'+onto)
    for key, value in phenotype.items():
        print(key, '-', value)
            #print("\tmatch type: " + annotation["matchType"])
            #global count, abstract, phenotype
            #if word not in count.keys():
            #    count[word] = label
            #    t = 0
            #elif word in count.keys() and count[word] != label:
            #    t = 1
            #else:
            #    t = 0
            #abstract[t] = abstract[t].replace(word, label)
            #if label not in phenotype:
            #    phenotype.append(label)
        #abstract[0] = abstract[0].replace(label, label+'_'+onto)
        #if abstract[1]:
        #    abstract[1] = abstract[1].replace(label, label+'_'+onto)
