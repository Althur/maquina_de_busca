
import json

from argparse import ArgumentParser
from nltk.tokenize import sexpr_tokenize

import search_engine.repository as se

from nltk.metrics.distance import edit_distance
from nltk.stem import PorterStemmer




def distance_editor(word, vocab):
    
    band = 10e2
    new = ''
    ps = PorterStemmer()
    word = ps.stem(word)
    
    if word in vocab:    
    
        return word
    
    
    for w in vocab:    
        
        dist = edit_distance(word, w) 
        
        
        if dist == 0:
            
            new = word
            
            break
        
        
        if dist < band:
        
            band = dist
            new = w
            
            
    print(f'Você quer dizer "{new}" ?')
    print(f'Mostrando os resultados da busca por "{new}":')
    print("_"*100)
    
    
    return new



    
def busca_and(index, query, vocab):
    
    query_terms = query.strip().split()
    
    
    if len(query_terms) == 0:
        
        return {}
    

    initial_term = query_terms[0]
    edited_initial_term = distance_editor(initial_term, vocab) # corrected initial term?
    
    docids = set(index[edited_initial_term]) if edited_initial_term in index else set() 
    
    words = []
    ps = PorterStemmer()
    
    
    for term in query_terms[1:]:
        
        new_word = distance_editor(term, vocab)
        word_stem = ps.stem(new_word)
        
        result = set(index[word_stem]) if word_stem in index else set()
        
        
        words.append(word_stem)
        docids &= result
        

    return docids, words




def busca_docids(index, query, vocab):
    
    
    result = [q.strip().strip('()') for q in sexpr_tokenize(query)]
    docids = set()
    
    
    for subquery in result:
        
        res, words = busca_and(index, subquery, vocab)
        docids |= res

    return docids, words




def busca(corpus, repo, index, query):
    
    # Parsing da query.
    # Recuperar os ids de documento que contem todos os termos da query.
    
    vocab = []
    
    
    for text, word in repo.items():
        
        vocab += word
        
        
    vocab = set(vocab)
    docids, words = busca_docids(index, query, vocab)
    
    # Retornar os textos destes documentos.
    
    
    return docids, words




def ranking(rank_doc, docids, words): 
    
    ranked_docs = []  
    
    
    for doc_id in docids:
        
        total = 0
        
        
        for word in words:
            
            tf = rank_doc[doc_id][word] 
            idf = rank_doc['idf'][word] 
            total += tf * idf
            
            
        ranked_docs.append([doc_id, total])
        

    ranked_docs = sorted(ranked_docs, key= lambda x: x[1])

    
    
    return [doc_id for doc_id, _ in ranked_docs]




def main():
    
    parser = ArgumentParser()
    parser.add_argument('corpus', help='Arquivo do corpus')
    parser.add_argument('repo', help='Arquivo do repo.')
    parser.add_argument('index', help='Arquivo do index.')
    parser.add_argument('ranking_file', help='Arquivo do index ranquedo.') 
    parser.add_argument('num_docs',
                        help='Numero maximo de documentos a retornar', 
                        type=int)
    parser.add_argument('query', help='A query (entre aspas)')
    args = parser.parse_args()
    

    corpus = se.load_corpus(args.corpus)
    
    

    with open(args.repo, 'r') as file:
        repo = json.load(file)

    with open(args.index, 'r') as file:
        index = json.load(file)
    
    with open(args.ranking_file, 'r') as file:
        rank_doc = json.load(file)
        
        

    docids, words = busca(corpus, repo, index, args.query)
    docids_ranqueados = ranking(rank_doc, docids, words) 
    docs = [corpus[docid] for docid in docids_ranqueados[:args.num_docs]]
    
    

    print(" ")
    
    for doc in docs:
        
        print("*" * 100)
        print(" ")
        print(doc)
        
        

    print(f'Numero de resultados: {len(docids)}')
    
    
    
    

if __name__ == '__main__':
    
    main()
    
