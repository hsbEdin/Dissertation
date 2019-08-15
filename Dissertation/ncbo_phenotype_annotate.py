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

def print_annotations(annotations, text_to_annotate, get_class=True):
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

        print("Class details")
        print("\tid: " + class_details["@id"])
        print("\tprefLabel: " + class_details["prefLabel"])
        print("\tontology: " + class_details["links"]["ontology"])
        print("Annotation details")
        if 'HP' in class_details["links"]["ontology"]:
            onto = 'HP'
        else:
            onto = 'ASDPTO'


        for annotation in result["annotations"]:
            print("\tfrom: " + str(annotation["from"]))
            print("\tto: " + str(annotation["to"]))
            x = int(annotation["from"]) - 1
            y = int(annotation["to"])
            word = text_to_annotate[x:y]
            label = class_details["prefLabel"]
            label = label.replace(' ','_')
            print("\tWORD: " + word)
            print("\tmatch type: " + annotation["matchType"])
            global count, abstract, phenotype

            if word not in count.keys():
                count[word] = label
                t = 0
            elif word in count.keys() and count[word] != label:
                t = 1
            else:
                t = 0
            abstract[t] = abstract[t].replace(word, label)
            if label not in phenotype:
                phenotype.append(label)
        abstract[0] = abstract[0].replace(label, label+'_'+onto)
        if abstract[1]:
            abstract[1] = abstract[1].replace(label, label+'_'+onto)


        print("\n")
filename = 'dysiexiamerge2.txt'
count = {}
phenotype = []
abstract = ['0','0']
with open(filename, 'r') as file_to_read:
    while True:
        text_to_annotate = file_to_read.readline()
        #text_to_annotate = file_to_read.readline() # 整行读取数据
        if not text_to_annotate:
            break
            pass
        #print(text_to_annotate[1:5], t, text_to_annotate)
        #t += 1
#####text_to_annotate = "Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye."
        abstract[0] = text_to_annotate
        abstract[1] = text_to_annotate
# Annotate using the provided text
        annotations = get_json(REST_URL + "/annotator?text=" + urllib.parse.quote(text_to_annotate) + '&ontologies=HP,ASDPTO')

# Print out annotation details
        print_annotations(annotations, text_to_annotate)

# Annotate with hierarchy information
        #annotations = get_json(REST_URL + "/annotator?max_level=3&text=" + urllib.parse.quote(text_to_annotate))
        #print_annotations(annotations, text_to_annotate)

# Annotate with prefLabel, synonym, definition returned
        #annotations = get_json(REST_URL + "/annotator?include=prefLabel,synonym,definition&text=" + urllib.parse.quote(text_to_annotate))
        #print_annotations(annotations, text_to_annotate, False)

        if count:
            file = open('dysiexia.txt', 'a')
            file.write(abstract[0])
            file.close()
            if abstract[1] != text_to_annotate:
                file = open('dysiexia.txt', 'a')
                file.write(abstract[1])
                file.close()
            file = open('dysiexia.txt', 'a')
            file.write(text_to_annotate)
            file.close()
        count = {}
if phenotype:
    file = open('dysiexiaPhenotype.txt', 'a')
    for i in range(len(phenotype)):
        s = str(phenotype[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
