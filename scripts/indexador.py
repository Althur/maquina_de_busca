#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:02:40 2020

@author: arthurrizzo
"""

import json
import nltk
import math

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from argparse import ArgumentParser
from collections import defaultdict, Counter




def create_repo(corpus):
    
    '''
    Cria o repositorio.
    Args:
        corpus: dicionario que mapeia um docid para uma string contendo o
                documento completo.
    Returns:
        Um dicionário que mapeia docid para uma lista de tokens.
    '''
    
    repo = {}
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    
    for docid, string in corpus.items():
        
        tokens = nltk.word_tokenize(string)
        stemmed = [ps.stem(x) for x in tokens if not x in stop_words] 
        clean = [w for w in stemmed if w.isalpha()]
        repo[docid] = clean
        
        
    return repo




def create_index(repo):
    
    
    '''
    Indexa os documentos de um corpus.
    Args:
        repo: dicionario que mapeia docid para uma lista de tokens.
    Returns:
        O índice reverso do repositorio: um dicionario que mapeia token para
        lista de docids.
    '''
    
    
    indexed = defaultdict(set)

    for doc_id, words in repo.items():
        
        for word in words:
            
            indexed[word].add(doc_id)


    for key in indexed:
        
        indexed[key] = list(indexed[key])
        
        
    return indexed




def ranking(corpus):
    
    rank = {}
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    rank['total'] = {}
    
    for doc_id, doc in corpus.items():
        
        tokens = nltk.word_tokenize(doc)
        stemmed = [ps.stem(x) for x in tokens if not x in stop_words] 
        alpha = [w for w in stemmed if w.isalpha()]
        counter = Counter(alpha)
        rank[doc_id] = {}
        
        
        for key, value in counter.items():
            
            rank[doc_id][key] = (1+math.log(value,2))
            
            
            try:
                
                rank['total'][key] += value
                
                
            except KeyError:
                
                rank['total'][key] = value
                
                
    rank['idf'] = {}
    
    
    for key, value in rank['total'].items():
        
        idf = math.log(len(corpus)/value, 2)
        rank['idf'][key] = idf


    return rank




def main():
    
    parser = ArgumentParser()
    
    
    parser.add_argument('corpus',
                        help='Arquivo json com um dicionario docid para texto')
    parser.add_argument('repo_name',
                        help='Raiz do nome do arquivo de repositorio')
    parser.add_argument('rank',
                        help='Arquivo json com os tfs e idfs')
    
    
    args = parser.parse_args()
    

    with open(args.corpus, 'r') as file_corpus:
        corpus = json.load(file_corpus)
        

    repo = create_repo(corpus)
    index = create_index(repo)
    rank = ranking(corpus)
    

    with open(args.repo_name + '_repo.json', 'w') as file_repo:
        json.dump(repo, file_repo, indent=4)

    with open(args.repo_name + '_index.json', 'w') as file_index:
        json.dump(index, file_index, indent=4)

    with open(args.rank + '_rank.json', 'w') as file_ranking:
        json.dump(rank, file_ranking, indent=4)
 
       
        

if __name__ == '__main__':
    
    main()
    
    
    
    