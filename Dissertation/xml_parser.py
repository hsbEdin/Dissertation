import os
import sys
import gzip
import xml.etree.ElementTree as ET

def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
sys.path.append('..')
# open and read gzipped xml files
pre_filename = '/afs/inf.ed.ac.uk/group/project/biomednlp/pubmed/data/pubmed19n0'
post_filename = '.xml.gz'

index = ''
filename = ''
folder = []
for i in range(1, 21):
    if i in range(1, 10):
        index = '00' + str(i)
        filename = pre_filename + index + post_filename
        folder.append(filename)
    elif i in range (10, 100):
        index = '0' + str(i)
        filename = pre_filename + index + post_filename
        folder.append(filename)
    else:
        index = str(i)
        filename = pre_filename + index + post_filename
        folder.append(filename)

# parse xml files
for file in folder:
    # open and read
    input_file = gzip.open(file)
    content = input_file.read()
    # parse
    root = ET.fromstring(content)
    # extract certain features of all articles
    articles = []
    for MedlineCitation in root.findall('./PubmedArticle/MedlineCitation'):
        article = {}

        for PMID in MedlineCitation.findall('PMID'):
            article[PMID.tag] = PMID.text

        for Article in MedlineCitation.findall('Article'):
            article['ArticleTitle'] = Article.find('ArticleTitle').text

        text = ''
        for AbstractText in MedlineCitation.findall('Article/Abstract/AbstractText'):
            if AbstractText.text != None:
                text += AbstractText.text
            article['Abstract'] = text

        for MeshHeading in MedlineCitation.findall('MeshHeadingList/MeshHeading'):

            for DescriptorName in MeshHeading.findall('DescriptorName'):
                if DescriptorName != None:
                    article.setdefault('MeshDescriptor', [])
                    if DescriptorName.text not in article['MeshDescriptor']:

                        article['MeshDescriptor'].append(DescriptorName.text)

            for QualifierName in MeshHeading.findall('QualifierName'):
                if QualifierName != None:
                    article.setdefault('MeshQualifier', [])
                    if QualifierName.text not in article['MeshQualifier']:
                        article['MeshQualifier'].append(QualifierName.text)


        for Keyword in MedlineCitation.findall('KeywordList/Keyword'):
            if Keyword != None:
                article.setdefault('Keyword', []).append(Keyword.text)

        for SupplMeshName in MedlineCitation.findall('SupplMeshList/SupplMeshName'):
            if SupplMeshName != None:
                article.setdefault('SupplMeshName', []).append(SupplMeshName.text)

        articles.append(article)

    root = ET.Element('ArticleList')
    tree = ET.ElementTree(root)

    for article in articles:
        father = ET.Element('Article')
        root.append(father)

        for k, v in article.items():

            if type(v) != list:
                child1 = ET.Element(k)
                child1.text = v
                father.append(child1)

            else:
                child1_name = k + 'List'
                child1 = ET.Element(child1_name)
                father.append(child1)
                for i in v:
                    grandchild1 = ET.Element(k)
                    grandchild1.text = i
                    child1.append(grandchild1)
    path = os.getcwd()
    indent(root, 0)
    output = file.split("/", 8)
    #print(output)
    #print(output[8])
    output1 = output[8].split(".", 1)
    output_file =  path+'/pubmedLite/'+output1[0] + 'Lite.xml'
    tree.write(output_file, 'UTF-8')
    input_file.close()
