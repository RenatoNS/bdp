# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:50:30 2020

@author: renatons
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import pickle
import os
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
import re
from gensim.models import Word2Vec, FastText
import umap
import hdbscan
from tqdm import tqdm
import unidecode
from scipy.stats import variation
import collections
import gc


# %% FUNÇÕES

def descobre_arquivos_na_pasta(pasta, tipo_do_arquivo='.xlsx'):
    # Descobre arquivos na pasta:
    arquivos = []
    for file in os.listdir(pasta):
        arquivos.append(os.fsdecode(file))
    arquivos = [arquivo for arquivo in arquivos if tipo_do_arquivo in arquivo]  # seleciona soh arquivos com .xlsx
    return arquivos



def get_nome_representacao_do_grupo_v5(df2, qtd_palavras, percentual_pra_manter_palavra_na_representacao, unidades):
    # pega cada palavra e ve as que mais se repetem nas sentences
    # fica com aquelas que estao em mais do que X% das sentences
    sentences = [sent for sent in df2['DS_ITEM_CLEAN']]
    if len(set(sentences)) == 1:  # ou seja, todos os itens sao iguais.
        representacao_grupo = [word for word in set(sentences)][0].split()
    else:
        palavras_series = df2['DS_ITEM_CLEAN'].str.split()
        palavras_series = palavras_series.apply(lambda x: x[:qtd_palavras])
        contagem_palavras_nas_sentences = {}
        # palavras = set([item for sublist in sentences for item in sublist.split()[:qtd_palavras]])
        palav = [item for sublist in palavras_series for item in sublist]
        palavras = sorted(set(palav),
                          key=palav.index)  # pra preservar a ordem em que as palavras aparecem (set puro coloca em ordem alfabetica)
        for palavra in palavras:
            contagem_palavras_nas_sentences[palavra] = (palavras_series.apply(lambda x: palavra in x)).sum()

        # contagem_palavras_nas_sentences = Series(contagem_palavras_nas_sentences).sort_values(ascending=False)
        contagem_palavras_nas_sentences = Series(contagem_palavras_nas_sentences)
        contagem_palavras_nas_sentences = contagem_palavras_nas_sentences / len(df2)
        contagem_palavras_nas_sentences = contagem_palavras_nas_sentences[
            contagem_palavras_nas_sentences > percentual_pra_manter_palavra_na_representacao]
        representacao_grupo = list(contagem_palavras_nas_sentences.index)
    # reordenacao:
    primeiras_palavras = [word for word in representacao_grupo if ((not word.isdigit()) and word not in unidades)]
    meio_palavras = [word for word in representacao_grupo if word.isdigit()]
    # ultimas_palavras = [word for word in representacao_grupo if ((word in unidades) and (word != 'x'))]
    ultimas_palavras = [word for word in representacao_grupo if word in unidades]
    # intercala numeros e unidades:
    if len(meio_palavras) == len(ultimas_palavras):
        result = [None] * (len(meio_palavras) + len(ultimas_palavras))
        result[::2] = meio_palavras
        result[1::2] = ultimas_palavras
        meio_palavras = result
        ultimas_palavras = []
    else:
        # if 'x' in representacao_grupo:
        if (('x' in ultimas_palavras) and (len(meio_palavras) > 0)):
            ultimas_palavras = [word for word in ultimas_palavras if word != 'x']  # retira o 'x', vai inserir abaixo:
            meio_palavras.insert(1, 'x')  # insere o 'x' apos o 1o numero
    if len(meio_palavras) == 0:
        representacao_grupo = primeiras_palavras  # daih nao coloca unidades
    else:
        representacao_grupo = primeiras_palavras + meio_palavras + ultimas_palavras

    df2['DS_ITEM_CORTE'] = df2['DS_ITEM_CLEAN'].apply(lambda x: x.split())
    df2['DS_ITEM_CORTE'] = df2['DS_ITEM_CORTE'].apply(lambda x: x[:qtd_palavras])

    mais_repetido = df2['DS_ITEM_CORTE'].value_counts().index[0]

    return (representacao_grupo, mais_repetido)



def print_qtd_grupos_sentencas_uteis(df, grupos, grupox):
    print('% de sentencas uteis:', (df[grupox] >= 0).sum() / len(df))
    print('qtd de grupos:', len(grupos))



def print_exemplos_grupos(df, inicio, fim, grupos, grupox, cols, qtd_palavras,
                          percentual_pra_manter_palavra_na_representacao, unidades):
    for grupo in grupos[inicio:fim]:
        df_mostrar = df[df[grupox] == grupo]
        print('\nGrupo:', grupo, 'len:', len(df_mostrar))
        print(get_nome_representacao_do_grupo_v4(df2=df[df[grupox] == grupo], qtd_palavras=qtd_palavras,
                                                 percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,
                                                 unidades=unidades))
        print(df_mostrar[cols])



def print_exemplos_grupos_v2_aleatorio(df, qtd_grupos_mostrar, grupos, grupox, cols, qtd_palavras,
                                       percentual_pra_manter_palavra_na_representacao, unidades):
    inicio = np.random.randint(len(grupos) - qtd_grupos_mostrar - 1)
    fim = inicio + qtd_grupos_mostrar
    for grupo in grupos[inicio:fim]:
        df_mostrar = df[df[grupox] == grupo]
        print('\nGrupo:', grupo, 'len:', len(df_mostrar))
        print(get_nome_representacao_do_grupo_v4(df2=df[df[grupox] == grupo], qtd_palavras=qtd_palavras,
                                                 percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,
                                                 unidades=unidades))
        print(df_mostrar[cols])
        
        
        
def get_nome_representacao_do_grupo_v4(df2, qtd_palavras, percentual_pra_manter_palavra_na_representacao, unidades):
    # pega cada palavra e ve as que mais se repetem nas sentences
    # fica com aquelas que estao em mais do que X% das sentences
    sentences = [sent for sent in df2['DS_ITEM_CLEAN']]
    if len(set(sentences)) == 1:  # ou seja, todos os itens sao iguais.
        representacao_grupo = [word for word in set(sentences)][0].split()
    else:
        palavras_series = df2['DS_ITEM_CLEAN'].str.split()
        palavras_series = palavras_series.apply(lambda x: x[:qtd_palavras])
        contagem_palavras_nas_sentences = {}
        # palavras = set([item for sublist in sentences for item in sublist.split()[:qtd_palavras]])
        palav = [item for sublist in palavras_series for item in sublist]
        palavras = sorted(set(palav),
                          key=palav.index)  # pra preservar a ordem em que as palavras aparecem (set puro coloca em ordem alfabetica)
        for palavra in palavras:
            contagem_palavras_nas_sentences[palavra] = (palavras_series.apply(lambda x: palavra in x)).sum()

        # contagem_palavras_nas_sentences = Series(contagem_palavras_nas_sentences).sort_values(ascending=False)
        contagem_palavras_nas_sentences = Series(contagem_palavras_nas_sentences)
        contagem_palavras_nas_sentences = contagem_palavras_nas_sentences / len(df2)
        contagem_palavras_nas_sentences = contagem_palavras_nas_sentences[
            contagem_palavras_nas_sentences > percentual_pra_manter_palavra_na_representacao]
        representacao_grupo = list(contagem_palavras_nas_sentences.index)
    # reordenacao:
    primeiras_palavras = [word for word in representacao_grupo if ((not word.isdigit()) and word not in unidades)]
    meio_palavras = [word for word in representacao_grupo if word.isdigit()]
    # ultimas_palavras = [word for word in representacao_grupo if ((word in unidades) and (word != 'x'))]
    ultimas_palavras = [word for word in representacao_grupo if word in unidades]
    # intercala numeros e unidades:
    if len(meio_palavras) == len(ultimas_palavras):
        result = [None] * (len(meio_palavras) + len(ultimas_palavras))
        result[::2] = meio_palavras
        result[1::2] = ultimas_palavras
        meio_palavras = result
        ultimas_palavras = []
    else:
        # if 'x' in representacao_grupo:
        if (('x' in ultimas_palavras) and (len(meio_palavras) > 0)):
            ultimas_palavras = [word for word in ultimas_palavras if word != 'x']  # retira o 'x', vai inserir abaixo:
            meio_palavras.insert(1, 'x')  # insere o 'x' apos o 1o numero
    if len(meio_palavras) == 0:
        representacao_grupo = primeiras_palavras  # daih nao coloca unidades
    else:
        representacao_grupo = primeiras_palavras + meio_palavras + ultimas_palavras
    return representacao_grupo


    
def get_nome_subgrupo(df, qtd_palavras2, percentual_pra_manter_palavra_na_representacao, unidades):
    
    col_edit = [" ".join(sent) for sent in df['DS_ITEM_CORTE']]
    df['DS_ITEM_SEG_CORTE'] = col_edit

    
    sentences2 = [" ".join(sent) for sent in df['DS_ITEM_SEG_CORTE']]
    if len(set(sentences2)) == 1:  # ou seja, todos os itens sao iguais.
        representacao_grupo2 = [word for word in set(sentences2)][0].split()
    else:
        palavras_series = df['DS_ITEM_SEG_CORTE'].str.split()
        palavras_series = palavras_series.apply(lambda x: x[:qtd_palavras2])
        contagem_palavras_nas_sentences = {}
        palav = [item for sublist in palavras_series for item in sublist]
        palavras = sorted(set(palav), key=palav.index)  # pra preservar a ordem em que as palavras aparecem (set puro coloca em ordem alfabetica)
        
        for palavra in palavras:
            contagem_palavras_nas_sentences[palavra] = (palavras_series.apply(lambda x: palavra in x)).sum()
    
        contagem_palavras_nas_sentences = Series(contagem_palavras_nas_sentences)
        contagem_palavras_nas_sentences = contagem_palavras_nas_sentences / len(df)
        contagem_palavras_nas_sentences = contagem_palavras_nas_sentences[contagem_palavras_nas_sentences > percentual_pra_manter_palavra_na_representacao]
        representacao_grupo2 = list(contagem_palavras_nas_sentences.index)
    # reordenacao:
    primeiras_palavras = [word for word in representacao_grupo2 if ((not word.isdigit()) and word not in unidades)]
    meio_palavras = [word for word in representacao_grupo2 if word.isdigit()]
    ultimas_palavras = [word for word in representacao_grupo2 if word in unidades]
    # intercala numeros e unidades:
    if len(meio_palavras) == len(ultimas_palavras):
        result = [None] * (len(meio_palavras) + len(ultimas_palavras))
        result[::2] = meio_palavras
        result[1::2] = ultimas_palavras
        meio_palavras = result
        ultimas_palavras = []
    else:        # if 'x' in representacao_grupo:
        if (('x' in ultimas_palavras) and (len(meio_palavras) > 0)):
            ultimas_palavras = [word for word in ultimas_palavras if word != 'x']  # retira o 'x', vai inserir abaixo:
            meio_palavras.insert(1, 'x')  # insere o 'x' apos o 1o numero
    if len(meio_palavras) == 0:
        representacao_grupo2 = primeiras_palavras  # daih nao coloca unidades
    else:
        representacao_grupo2 = primeiras_palavras + meio_palavras + ultimas_palavras
    
    df['DS_ITEM_SEG_CORTE'] = df['DS_ITEM_SEG_CORTE'].apply(lambda x: x.split())
    df['DS_ITEM_SEG_CORTE'] = df['DS_ITEM_SEG_CORTE'].apply(lambda x: x[:qtd_palavras2])
    
    mais_repetido = df['DS_ITEM_SEG_CORTE'].value_counts().index[0]
    
    return (representacao_grupo2, mais_repetido)


def get_nome_grupo_filho(df, qtd_palavras2, percentual_pra_manter_palavra_na_representacao, unidades):
    
    col_edit2 = [" ".join(sent) for sent in df['DS_ITEM_SEG_CORTE']]
    df['DS_ITEM_TER_CORTE'] = col_edit2


    sentences2 = [" ".join(sent) for sent in df['DS_ITEM_TER_CORTE']]
    if len(set(sentences2)) == 1:  # ou seja, todos os itens sao iguais.
        representacao_grupo2 = [word for word in set(sentences2)][0].split()
    else:
        palavras_series = df['DS_ITEM_TER_CORTE'].str.split()
        palavras_series = palavras_series.apply(lambda x: x[:qtd_palavras2])
        contagem_palavras_nas_sentences = {}
        palav = [item for sublist in palavras_series for item in sublist]
        palavras = sorted(set(palav), key=palav.index)  # pra preservar a ordem em que as palavras aparecem (set puro coloca em ordem alfabetica)
        
        for palavra in palavras:
            contagem_palavras_nas_sentences[palavra] = (palavras_series.apply(lambda x: palavra in x)).sum()
    
        contagem_palavras_nas_sentences = Series(contagem_palavras_nas_sentences)
        contagem_palavras_nas_sentences = contagem_palavras_nas_sentences / len(df)
        contagem_palavras_nas_sentences = contagem_palavras_nas_sentences[contagem_palavras_nas_sentences > percentual_pra_manter_palavra_na_representacao]
        representacao_grupo2 = list(contagem_palavras_nas_sentences.index)
    # reordenacao:
    primeiras_palavras = [word for word in representacao_grupo2 if ((not word.isdigit()) and word not in unidades)]
    meio_palavras = [word for word in representacao_grupo2 if word.isdigit()]
    ultimas_palavras = [word for word in representacao_grupo2 if word in unidades]
    # intercala numeros e unidades:
    if len(meio_palavras) == len(ultimas_palavras):
        result = [None] * (len(meio_palavras) + len(ultimas_palavras))
        result[::2] = meio_palavras
        result[1::2] = ultimas_palavras
        meio_palavras = result
        ultimas_palavras = []
    else:        # if 'x' in representacao_grupo:
        if (('x' in ultimas_palavras) and (len(meio_palavras) > 0)):
            ultimas_palavras = [word for word in ultimas_palavras if word != 'x']  # retira o 'x', vai inserir abaixo:
            meio_palavras.insert(1, 'x')  # insere o 'x' apos o 1o numero
    if len(meio_palavras) == 0:
        representacao_grupo2 = primeiras_palavras  # daih nao coloca unidades
    else:
        representacao_grupo2 = primeiras_palavras + meio_palavras + ultimas_palavras
    
    df['DS_ITEM_TER_CORTE'] = df['DS_ITEM_TER_CORTE'].apply(lambda x: x.split())
    df['DS_ITEM_TER_CORTE'] = df['DS_ITEM_TER_CORTE'].apply(lambda x: x[:qtd_palavras2])
    
    mais_repetido = df['DS_ITEM_TER_CORTE'].value_counts().index[0]
    
    return (representacao_grupo2, mais_repetido)



# %%

##################
#TREINO DO MODELO:
##################

import os
folder = r'C:/Users/RenatoNS/Desktop/banco_de_precos_novo/data/'     ### Inserir aqui a pasta local do repositorio
os.chdir(folder)
cols_mostrar = ['DS_ITEM_CLEAN','UnidadeAgrupada']


# %% PARAMETROS

tamanho_minimo_pra_formar_grupo = 2   ### 30 -------> meu =2
qtd_palavras = 16
qtd_dimensoes = 100   ### 300
qtd_dimensoes_umap = 15
quantile_a_retirar_outliers_dbscan = 0.95
minimo_cosine_similarity = 0.9
qtd_ngram_FastText = 3
cv_maximo_pra_considerar_grupo_homogeneo_sentenca = 1.75
percentual_primeira_palavra_igual_pra_considerar_grupo_homogeneo = 0.50
percentual_pra_manter_palavra_na_representacao = 0.50
qtd_min_auditadas_para_formar_grupo = 1  ### 10 -----------> meu = 1
quantile_a_retirar_numeros_diferentes_no_grupo = 0.95
quantile_a_retirar_quantidade_palavras_diferentes_no_grupo = 0.95   ###------0.8

qtd_palavras2 = 3
qtd_palavras3 = 1


# %%

# Descobre arquivos de dados:
pasta = 'C:/Users/RenatoNS/Desktop/banco_de_precos_novo/data/'           ### Inserir aqui a pasta local do repositorio
arquivos = descobre_arquivos_na_pasta(pasta,tipo_do_arquivo='.xlsx')


# %% LEITURA INICIAL - demorado - le arquivos e concatena registros em um dataframe:

lista = []
for arquivo in tqdm(arquivos):
    df = pd.read_excel(pasta + arquivo, dtype={'DS_ITEM':str,'UnidadeAgrupada':str})
    lista.append(df)
licitacon = pd.concat(lista,ignore_index=True)

#%% LIMPEZA DE CARACTERES ESPECIAIS DE DS_ITEM

#p = re.compile(r'[^\w\s]+')
#df['DS_ITEM'] = [p.sub('', x) for x in df['DS_ITEM'].tolist()]

df['DS_ITEM'] = df['DS_ITEM'].str.replace(r'[^\w\s]+', '').astype(str)


#%% REDUÇÃO DO DF PARA TESTE

df_teste = df.iloc[:90000,:]
df = df_teste

licitacon = df_teste


#%%

licitacon = licitacon.set_index('codigoitemNFE')

#deixa somente colunas necessarias:
licitacon = licitacon[['NCMSH', 'DS_ITEM', 'UnidadeAgrupada']]


# %% Pra fazer download dos corpus/stopwords:

#nltk.download()
#nltk.download('punkt')


# %%

# stopwords sao somente punctuation. O resto DEIXO, tem palavras importantes pros produtos: com/sem/tem/nem, etc.
stopwords = set(list(punctuation))
stopwords2 = ['da', 'de', 'do']

unidades = ['x','mm','m','cm','ml','g','mg','kg','unidade','unidades','polegada','polegadas','grama','gramas','gb',
            'mb','l','litro','litros','mts','un','mgml','w','hz','v','gr','lt','lts','lonas','cores','mcg']
primeira_palavra_generica = ['caixa','jogo','kit','conjunto','item','it','cjt','conj','conjt','jg','kt','de','para']


# %% DATA CLEAN

#limpa sentencas retirando stopwords (tem que ser minusculo) e pontuacao.
licitacon['DS_ITEM_CLEAN'] = [ ' '.join([word for word in item.split() if word.lower() not in stopwords]) for item in licitacon['DS_ITEM'].astype(str) ]
licitacon['DS_ITEM_CLEAN'] = [ ' '.join([word for word in item.split() if word.lower() not in stopwords2]) for item in licitacon['DS_ITEM'].astype(str) ]

#insere espaco apos / e -, pra no final nao ficar palavras assim: csolucao, ptexto (originais eram c/solucao, p-texto)
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: re.sub(r'/|-',r' ',x))
#retira pontuacao, /, etc:
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
#passa pra minusculo:
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: x.lower())
#insere espaco apos numero e letra (separa unidades de medida:) ex.: 500ml vs 100ml vs 500mg
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: re.sub(r'(\d{1})(\D)',r'\1 \2',x))
#insere espaco apos letra e numero ex.:c100 pc50
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: re.sub(r'(\D{1})(\d)',r'\1 \2',x))
#retira espacos duplicados
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: re.sub(r' +',r' ',x))
#retira acentos:
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: unidecode.unidecode(x))
#se primeira palavra for numero, joga pro final (caso de numeros de referencia que colocam no inicio)
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: ' '.join(x.split()[1:] + [x.split()[0]]) if ((len(x) > 1) and (x.split()[0].isdigit()) ) else x)
# remove zeros a esquerda de numeros (02 litros, 05, etc.)
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: ' '.join([word.lstrip('0') for word in x.split()] ) )
# remove 'x', pra não diferenciar pneu 275 80 de 275 x 80:
licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: ' '.join([word for word in x.split() if word is not 'x']))


# %%

#retira primeira palavra se estah em unidades ou primeira_palavra_generica:
# roda varias vezes pra tirar todas as primeiras palavras:
for _ in range(3):
    licitacon['DS_ITEM_CLEAN'] = licitacon['DS_ITEM_CLEAN'].apply(lambda x: ' '.join(x.split()[1:]) if (len(x) > 1 and (x.split()[0] in unidades or x.split()[0] in primeira_palavra_generica)) else x )


# %%

#a partir de agora filtra soh pra material de consumo OU Equipamentos e Material Permanente:
df = licitacon.copy()

# retira livros:
df = df[df['DS_ITEM_CLEAN'].apply(lambda x: 'livro' not in x)]


# %%

#limito a 16 palavras - pega 85% dos itens até 16 palavras, o resto é lixo, tem item com mais de 300 palavras.
sentences = [sent.split()[:qtd_palavras] for sent in df['DS_ITEM_CLEAN']]


# %% WORD2VEC

print('Treino word2vec/fastText word embeddings gensim:')
model = FastText(sentences,size=qtd_dimensoes, min_count=tamanho_minimo_pra_formar_grupo, workers=-1, min_n=qtd_ngram_FastText, max_n=qtd_ngram_FastText, iter=10)


# %%

print('Conversao word embeddings to sentence embedding, com pesos:')
doc_vectors = {}
for number, sent in enumerate(tqdm(sentences)):
    # dando peso maior pras primeiras palavras, peso decrescente ateh o final, numeros com mesmo peso da primeira palavra:
    if len(sent) == 0:
        doc_vectors[number] = np.zeros(qtd_dimensoes,)
    elif len(sent) == 1:
        doc_vectors[number] = model[sent[0]]
    elif len(sent) > 1:
        pesos = np.array(range(len(sent))[::]) + 1
        pesos = 1 / pesos # agora com pesos 1/x - tem que ser na ordem 1,2,..., os menores numeros dao maiores pesos - decai menos que exponencial, que eh muito brusca a queda.
        media = []
        divisao = 0
        counter = 0
        for word in sent:
            if word.isdigit():
                # Nova abordagem: se eh digit, atribui peso NO 3/4 da faixa entre o peso da primeira e da ultima palavra.
                # Mesmo peso pra todos os numeros, mais importante que palavras do fim, menos importante que palavras do inicio.
                media.append(model.wv[word] * ((pesos[0]+pesos[-1])*(1/4)) ) 
                divisao += ((pesos[0]+pesos[-1])*(1/4))
            else:
                media.append(model.wv[word] * pesos[counter])
                divisao += pesos[counter]
            counter += 1
        doc_vectors[number] = np.array(media).sum(axis=0) / divisao #media de tudo

doc_vectors = DataFrame(doc_vectors).T
doc_vectors = doc_vectors.set_index(df.index) # coloca o codigoitemNFE


# %%

print('StandardScaler:')
scaler = StandardScaler()
doc_vectors_std_df = DataFrame(scaler.fit_transform(doc_vectors),index=doc_vectors.index,columns=doc_vectors.columns)


#%% UMAP(DEMORA)

# Reduz dims com UMAP - DEMORA - Reduz pra 15 dimns com UMAP direto das 300 dimns, sem PCA:
print('Reduz com UMAP pra', str(qtd_dimensoes_umap),'dimensoes.')
# agora com metric = 'cosine'. desempenho colocando init='random' piora, mas eh mais rapido.
umap_redux = umap.UMAP(n_components=qtd_dimensoes_umap, random_state=999, metric='cosine',verbose=True)
umap_redux.fit(doc_vectors_std_df) 

doc_vectors_std_df_umap = umap_redux.transform(X=doc_vectors_std_df)


# %% HDBSCAN clustering(DEMORA)

print('Clusterizando, tamanho minimo pra formar grupo:', str(tamanho_minimo_pra_formar_grupo))
min_samples = 1
clustering = hdbscan.HDBSCAN(min_cluster_size=tamanho_minimo_pra_formar_grupo,min_samples=min_samples,prediction_data=True,core_dist_n_jobs=-1)
clustering.fit(doc_vectors_std_df_umap)

df['grupo'] = clustering.labels_


# %%

# atribui -2 aos outliers pelo HDBSCAN:
threshold = pd.Series(clustering.outlier_scores_).quantile(quantile_a_retirar_outliers_dbscan)
outliers = np.where(clustering.outlier_scores_ > threshold)[0]
df.iloc[outliers,df.columns.get_loc('grupo')] = -2

grupos = np.unique(clustering.labels_)
grupos = [grupo for grupo in grupos if grupo >= 0]

print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo')


# %% DEMORA

#########################################
# EXCLUI sentences outliers pelo texto pelos exemplars - DEMORA
# FAZ cosine distance pra cada sentence dentro dos grupos
#########################################

print('EXCLUI sentences outliers pelo texto pelos exemplars:')

#pontos mais representativos de cada grupo (como hdbscan aceita formato aleatorio de cada cluster, nao tem como passar/nao existe um centroide, como no k-means, que assume grupos esfericos)
exemplars = []
for exemplar in tqdm(clustering.exemplars_):
    exemplars.append(np.mean(exemplar,axis=0))
exemplars_df = DataFrame(exemplars,index=range(len(grupos)))

map_grupos_exemplars = {}
df_temp = DataFrame(columns=['sims'])

for grupo in tqdm(grupos[:]):

    df2 = df[df['grupo'] == grupo]
    indexes = df2.index
    grupo_vectors = DataFrame(doc_vectors_std_df_umap,index=df.index).loc[indexes]
    
    grupo_do_exemplar = Series(cosine_similarity(grupo_vectors.mean(axis=0).values.reshape(1,-1),exemplars_df)[0]).sort_values(ascending=False).index[0]
    map_grupos_exemplars[grupo] = grupo_do_exemplar

    sims = cosine_similarity(grupo_vectors,exemplars[grupo_do_exemplar].reshape(1,-1))

    df2['sims'] = sims
    df_temp = df_temp.append(df2[['sims']])

#passa resultados pro df principal:
df['sims'] = df_temp
df['sims'] = df['sims'].replace(np.nan,-1)

#retira quem tem similaridade negativa - eh um bom parametro.
df['grupo2'] = np.where(df['sims'] < 0, -1, df['grupo'])

grupos = df['grupo2'].unique()
grupos = [grupo for grupo in grupos if grupo >= 0]

print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo2')

#######################
# Reduzi um passo aqui:
df['grupo3'] = df['grupo2']


# %%

##########################################
# Se a 1a palavra nao for a mesma em X% do grupo, exclui o grupo, eh muito heterogeneo: - eh RAPIDO
##########################################

print('Se a 1a palavra nao for a mesma em X% do grupo, exclui o grupo, eh muito heterogeneo:')

grupos = sorted(df['grupo3'].unique())
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

grupos_homogeneos = []

for grupo in tqdm(grupos):
    df2 = df[df['grupo3'] == grupo]
    if len(df2) > 0:
        if ( df2['DS_ITEM_CLEAN'].apply(lambda x: x.split()[0] if (len(x.split()) > 0) else np.random.random()).value_counts().iloc[0] / len(df2) ) > percentual_primeira_palavra_igual_pra_considerar_grupo_homogeneo:
            grupos_homogeneos.append(grupo)

df['grupo4'] = df['grupo3'].isin(grupos_homogeneos)
df['grupo4'] = np.where(df['grupo4'], df['grupo3'], -1)

grupos = sorted(df['grupo4'].unique())
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo4')


# %%

############################################################################
# EXCLUI GRUPOS NAO HOMOGENEOS - pela contagem de palavras diferentes! - eh RAPIDO
############################################################################

df['qtd_palavras_diferentes'] = df['DS_ITEM_CLEAN'].apply(lambda x: len(set( [item for sublist in [sent.split()[:qtd_palavras] for sent in x] for item in sublist    ] ) ))
qtd_palavras_por_grupo = df.groupby('grupo4')['qtd_palavras_diferentes'].median()
qtd_palavras_por_grupo = qtd_palavras_por_grupo.sort_values()
qtd_max_palavras_diferentes_no_grupo = int(qtd_palavras_por_grupo.quantile(quantile_a_retirar_quantidade_palavras_diferentes_no_grupo))
print('Quantidade maxima de palavras diferentes aceita por grupo:', qtd_max_palavras_diferentes_no_grupo)

df['qtd_median_palavras_dif_grupo'] = df['grupo4'].map(qtd_palavras_por_grupo)

df['grupo5'] = np.where(df['qtd_median_palavras_dif_grupo'] > qtd_max_palavras_diferentes_no_grupo, -1, df['grupo4'])

grupos = df['grupo5'].unique()
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo5')



# %% ***CASO NAO SEJA FEITA A LIMPEZA DAS LINHAS 332 ATE 339, DA ERRO AQUI***

############################################################################
# EXCLUI GRUPOS que tem items SOMENTE DE UMA MESMA AUDITADA - sao escritos muito especificos, queremos coisas compradas por varias auditadas. - eh RAPIDO
############################################################################

qtd_auditadas_por_grupo = df.groupby(['grupo5'])['DS_ITEM'].apply(lambda x: len(np.unique(x)) )
df['qtd_auditadas_diferentes_do_grupo'] = df['grupo5'].map(qtd_auditadas_por_grupo)
df['grupo6'] = np.where(df['qtd_auditadas_diferentes_do_grupo'] > qtd_min_auditadas_para_formar_grupo, df['grupo5'], -1)


# %%

grupos = df['grupo6'].unique()
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.

print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo6')


# %%

############################################################################
# EXCLUI GRUPOS que tem MUITOS NUMEROS DIFERENTES NAS SENTENCES: - RAPIDO 
############################################################################

df['qtd_numeros_diferentes'] = df['DS_ITEM_CLEAN'].apply(lambda x: len(set( [item for sublist in [sent.split()[:qtd_palavras] for sent in x] for item in sublist if item.isdigit()   ] ) ))
df['qtd_numeros_diferentes'] = np.where(df['qtd_numeros_diferentes'] == 0, 0, df['qtd_numeros_diferentes']-1)


# %%

qtd_numeros_por_grupo = df.groupby('grupo4')['qtd_numeros_diferentes'].median()
qtd_numeros_por_grupo = qtd_numeros_por_grupo[qtd_numeros_por_grupo > 0]
qtd_numeros_por_grupo = qtd_numeros_por_grupo.sort_values()


# %%

qtd_max_numeros_diferentes_no_grupo = int(qtd_numeros_por_grupo.quantile(quantile_a_retirar_numeros_diferentes_no_grupo))
print('Quantidade maxima de numeros diferentes aceita por grupo:', qtd_max_numeros_diferentes_no_grupo)


# %%

df['qtd_median_numeros_dif_grupo'] = df['grupo4'].map(qtd_numeros_por_grupo)

df['grupo7'] = np.where(df['qtd_median_numeros_dif_grupo'] > qtd_max_numeros_diferentes_no_grupo, -1, df['grupo6'])

grupos = df['grupo7'].unique()
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.


print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo7')


# %%

###########################
# passo reduzido:
###########################

df['grupo8'] = df['grupo7']

###########################
print('Conversao word embeddings to sentence embedding, com pesos:')
###########################

model2 = Word2Vec(sentences,size=qtd_dimensoes, min_count=1,workers=-1)

doc_vectors2 = {}

for number, sent in enumerate(tqdm(sentences)):
    #agora dando peso maior pras primeiras palavras, peso decrescente ateh o final, numeros com mesmo peso da primeira palavra:
    if len(sent) == 0:
        doc_vectors2[number] = np.zeros(qtd_dimensoes,)
    elif len(sent) == 1:
        doc_vectors2[number] = model2.wv[sent[0]]
    elif len(sent) > 1:
        pesos = np.array(range(len(sent))[::]) + 1
        pesos = 1 / pesos # agora com pesos 1/x - tem que ser na ordem 1,2,..., os menores numeros dao maiores pesos - decai menos que exponencial, que eh muito brusca a queda.
        media = []
        divisao = 0
        counter = 0
        for word in sent:
            # media.append(model2.wv[word])
            # divisao += 1
            ######### AGORA O MODEL EH W2V E O PESO EH DOBRADO PRA DIGITS:
            if word.isdigit():
                media.append(model2.wv[word] * ((pesos[0]+pesos[-1])*(1/2)) )
                divisao += ((pesos[0]+pesos[-1])*(1/2))
            else:
                media.append(model2.wv[word] * pesos[counter])
                divisao += pesos[counter]
            counter += 1
        doc_vectors2[number] = np.array(media).sum(axis=0) / divisao #media de tudo

doc_vectors2 = DataFrame(doc_vectors2).T
doc_vectors2 = doc_vectors2.set_index(df.index)

doc_vectors_grupos = {}

for grupo in tqdm(grupos):
    indices = df[df['grupo8'] == grupo].index
    doc_vectors_grupos[grupo] = doc_vectors2.loc[indices]
    doc_vectors_grupos[grupo] = doc_vectors_grupos[grupo].mean(axis=0)

doc_vectors_grupos = DataFrame(doc_vectors_grupos).T


# %%

#usa o scaler original:
doc_vectors_grupos_std = DataFrame(scaler.transform(doc_vectors_grupos),index=doc_vectors_grupos.index,columns=doc_vectors_grupos.columns)

#%% ***QUASE TRAVA, COM 8GB DE RAM CHEGA A 95% DE USO***
grupos_similarities = cosine_similarity(doc_vectors_grupos_std)

#%% 
grupos_similarities = DataFrame(grupos_similarities,index=doc_vectors_grupos.index,columns=doc_vectors_grupos.index)

#%%
similarity_minima_pra_juntar_grupos = 0.90


# %% DEMORA 1:54h com 150k

#junta os grupos:
grupos_similares = []
for grupo in tqdm(grupos_similarities):
    agrupar_df = grupos_similarities[grupo].sort_values(ascending=False)
    agrupar_df = agrupar_df[agrupar_df >= similarity_minima_pra_juntar_grupos]
    grupos_similares.append(list(agrupar_df.index))


# %%

novo_grupo = 0
mapeamento_grupos = {}
for mini_grupo in tqdm(grupos_similares):
    if len(mini_grupo) == 1:
        mapeamento_grupos[mini_grupo[0]] = novo_grupo
    else:
        for grupo in mini_grupo:
            if grupo not in mapeamento_grupos.keys():
                for mini_grupo2 in grupos_similares:
                    if grupo in mini_grupo2:
                        mapeamento_grupos[grupo] = novo_grupo
                        for grupo2 in mini_grupo2:
                            if grupo2 not in mapeamento_grupos.keys():
                                mapeamento_grupos[grupo2] = novo_grupo
    novo_grupo += 1


# %%

df['grupo9'] = df['grupo8'].map(mapeamento_grupos)
df['grupo9'] = df['grupo9'].fillna(-1)
df['grupo9'] = df['grupo9'].astype(int)

grupos = df['grupo9'].unique()
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.


# %%

print_exemplos_grupos_v2_aleatorio(df=df,qtd_grupos_mostrar=10,grupos=grupos,grupox='grupo9',cols=cols_mostrar+['DS_ITEM'],
                                   qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)
print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo9')


# %%

###############################################
# Agora repassar as sentences do grupo -1 e tentar encaixá-las nos grupos formados, só com grande certeza. - Demora 5 min.
###############################################

excluidos_index =  df[df['grupo9'] == -1].index
incluidos_index =  df[df['grupo9'] >= 0].index

doc_vectors_excluidos = doc_vectors.loc[excluidos_index]
doc_vectors_incluidos = doc_vectors.loc[incluidos_index]

doc_vectors_grupos_finais = {}
for grupo in tqdm(grupos):
    df2 = df[df['grupo9'] == grupo]
    doc_vectors_grupos_finais[grupo] = doc_vectors_incluidos.loc[df2.index]
    doc_vectors_grupos_finais[grupo] = doc_vectors_grupos_finais[grupo].values.mean(axis=0)
# index aqui jah eh o numero certo dos grupos:
doc_vectors_grupos_finais = DataFrame(doc_vectors_grupos_finais).T


# %%

compara = cosine_similarity(doc_vectors_excluidos.loc[excluidos_index],doc_vectors_grupos_finais.values)
compara = DataFrame(compara,index=excluidos_index, columns=grupos)

similarity_do_grupo_mais_parecido = compara.max(axis=1)


# %% ### EM UM DF MUITO GRANDE, TRAVA AQUI ###

grupo_mais_parecido = compara.idxmax(axis=1)

# %%

similarity_minima_pra_encaixar_itens_excluidos_no_final = 0.95

encaixar_excluidos = Series( np.where(similarity_do_grupo_mais_parecido >= similarity_minima_pra_encaixar_itens_excluidos_no_final, grupo_mais_parecido, -1), index= similarity_do_grupo_mais_parecido.index)
df['grupo10'] = encaixar_excluidos
df['grupo10'] = df['grupo10'].fillna(-1)
df['grupo10'] = np.where(df['grupo10'] == -1, df['grupo9'], df['grupo10'])
    
grupos = df['grupo10'].unique()
grupos = [grupo for grupo in grupos if grupo >=0] #tirar os -1, -2, etc.


# %%
                
print_exemplos_grupos_v2_aleatorio(df=df,qtd_grupos_mostrar=10,grupos=grupos,grupox='grupo10',cols=cols_mostrar+['DS_ITEM'],
                                   qtd_palavras=qtd_palavras,percentual_pra_manter_palavra_na_representacao=percentual_pra_manter_palavra_na_representacao,unidades=unidades)

print_qtd_grupos_sentencas_uteis(df=df,grupos=grupos,grupox='grupo10')


# %%  CHAMADA DA FUNÇÃO QUE É RESPONSAVEL PELA CRIAÇÃO DA COLUNA DS_ITEM_CORTE, PRIMEIRA HERANÇA QUE ACEITA 16 PALAVRAS 

get_nome_representacao_do_grupo_v5(df, qtd_palavras, percentual_pra_manter_palavra_na_representacao, unidades)


# %% ESSA FUNÇÃO CRIA A SEGUNDA HERANÇA, QUE ACEITA 3 PALAVRAS

get_nome_subgrupo(df, qtd_palavras2, percentual_pra_manter_palavra_na_representacao, unidades)


# %% ESSA FUNÇÃO, DEFINE O NOME DA CLASSE PAI, ACEITA SOMENTE 1 PALAVRA 

get_nome_grupo_filho(df, qtd_palavras3, percentual_pra_manter_palavra_na_representacao, unidades)


# %% FUNÇÃO PARA SALVAR O CSV COM AS HERANÇAS DOS GRUPOS

df[['DS_ITEM', 'DS_ITEM_CORTE', 'DS_ITEM_SEG_CORTE', 'DS_ITEM_TER_CORTE']].to_csv(r'C:\Users\RenatoNS\Desktop\banco_de_precos_novo\data\subgrupos_teste.csv',sep=';', encoding="utf-8-sig")
