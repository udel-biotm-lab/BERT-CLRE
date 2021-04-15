from __future__ import print_function,division
import re,codecs,random
import numpy as np
import pandas as pd
import stanza
import networkx as nx
from eda import synonym_replacement,random_swap,random_deletion

class DataReader:
    def __init__(self,file_path):
        self.file_path=file_path
        self.line_list=[]
        self.sent_list=[]
    def read_file(self):
        count=0
        with codecs.open(self.file_path, encoding='utf8') as f:
            for line in f:
                self.line_list.append(line.strip())
                count+=1
                #if count>10:
                #    break
        return self.line_list
    def parse_line(self):
        for li in self.line_list:
            self.sent_list.append(li.split('\t'))
        return self.sent_list


class ExtractSDP:
    def __init__(self):
        self.nlp = stanza.Pipeline(lang='en') 

    def extract(self,sent,entities):
        
        entity1=None
        entity2=None
        doc = self.nlp(sent)

        #doc.sentences[0].print_dependencies()
        tokenized_sent=[]
        sent_num=len(doc.sentences)
        for si in range(sent_num):
            tokenized_sent+=[token.text for token in doc.sentences[si].tokens]

        edges = []
        for token in doc.sentences[0].dependencies:
            if token[0].text.lower() != 'root':
                edges.append(((token[0].text,token[0].id), (token[2].text,token[2].id)))
        
        #assume that entity1 always appear before entity2 and we also need the index of the entities
        for token in doc.sentences[0].tokens:
            if not entity1 and token.text == entities[0]:
                entity1=(token.text,token.id[0])
                continue
            if entity1 and token.text == entities[1]:
                entity2=(token.text,token.id[0])
        #print('entity1',entity1)
        #print('entity2',entity2)
        graph = nx.Graph(edges)
        # Get the length and path
        
        #print(nx.shortest_path_length(graph, source=entity1, target=entity2))
        try:
            sdp=nx.shortest_path(graph, source=entity1, target=entity2)
        except:
            sdp=[]
        if not entity1 or not entity2:
            sdp=[]
        return sdp, tokenized_sent

class edaRE:
    def __init__(self,file_path,aug_file_path,task_name):
        self.file_path = file_path
        self.aug_file_path=aug_file_path
        dr=DataReader(self.file_path)
        dr.read_file()
        sent_list=dr.parse_line()
        if task_name=='ppi':
            
            self.el_sdp=['PROTA','PROTB']
            self.el=['@PROTEIN$','@PROTEIN$']
            self.sent_list=[]
            for senti in sent_list:
                self.sent_list.append([senti[2],senti[1]])
        elif task_name=='ddi':
            sent_list=sent_list[1:]
            self.sent_list=[]
            for senti in sent_list:
                self.sent_list.append([senti[2],senti[1]])
            self.el_sdp=['DRUGA','DRUGB']
            self.el=['@DRUG$','@DRUG$']
        elif task_name=='chemprot':
            sent_list=sent_list[1:]
            self.sent_list=[]
            for senti in sent_list:
                self.sent_list.append([senti[2],senti[1]])
            self.el_sdp=['PROTA','CHEMA']
            self.el=['@GENE$','@CHEMICAL$']
        elif task_name=='mirgene':
            sent_list=sent_list[1:]
            self.sent_list=[]
            for senti in sent_list:
                self.sent_list.append([senti[1],senti[0]])
            self.el_sdp=['RNAA','GENE']
            self.el=['@RNA$','@GENE$']
        elif task_name=='ds':
            
            self.el_sdp=['PROTA','PROTB']
            self.el=['@PROTEIN$','@PROTEIN$']
            self.sent_list=[]
            for senti in sent_list:
                self.sent_list.append([senti[1],senti[0]])
        elif task_name=='ds-ddi':
            
            self.el_sdp=['DRUGA','DRUGB']
            self.el=['@DRUG$','@DRUG$']
            self.sent_list=[]
            for senti in sent_list:
                self.sent_list.append([senti[1],senti[0]])
        elif task_name=='ds-chemprot':
            
            self.el_sdp=['PROTA','CHEMA']
            self.el=['@PROTEIN$','@CHEMICAL$']
            self.sent_list=[]
            for senti in sent_list:
                self.sent_list.append([senti[1],senti[0]])
        elif task_name=='ds-mirgene':
            
            self.el_sdp=['RNAA','GENE']
            self.el=['@RNA$','@GENE$']
            self.sent_list=[]
            for senti in sent_list:
                self.sent_list.append([senti[1],senti[0]])
        self.esdp=ExtractSDP()

    def augment(self,sr_num,rw_num,rd_num):
        
        aug_sent_list=[]
        with codecs.open(self.aug_file_path, 'w+',encoding='utf8') as f:
            for senti in self.sent_list:
                si=senti[1].replace(self.el[0],self.el_sdp[0],1).replace(self.el[1],self.el_sdp[1],1) 
                print('Original sentence: ',si)
                reversed_el_sdp=[self.el_sdp[1],self.el_sdp[0]]
                sdp, tokenized_sent=self.esdp.extract(si,self.el_sdp)
                print('Shortest dependency path: ',sdp)
                print('Tokenized sentence: ',tokenized_sent)
                if len(sdp)==0:
                    sdp, tokenized_sent=self.esdp.extract(si,reversed_el_sdp)
                    print('New shortest dependency path: ',sdp)

                alpha_sr=0.1
                n_sr = max(1, int(alpha_sr*len(tokenized_sent)))
                sdp_words=[si[0] for si in sdp]
                for sri in range(sr_num):
                    sent_sr = synonym_replacement(tokenized_sent,sdp_words,n_sr)
                    sent_sr=sent_sr.replace(self.el_sdp[0],self.el[0],1).replace(self.el_sdp[1],self.el[1],1) 
                    print('Sentence with synonym replacement:',sent_sr)
                    
                    senti.append(sent_sr)
                
                sdp_word_idx=[si[1]-1 for si in sdp]
                for swi in range(rw_num):
                    sent_sw = random_swap(tokenized_sent,sdp_word_idx,n_sr)
                    sent_sw=sent_sw.replace(self.el_sdp[0],self.el[0],1).replace(self.el_sdp[1],self.el[1],1) 
                    print('Sentence with random swap:',sent_sw)
                    senti.append(sent_sw)
                for sdi in range(rd_num):
                    sent_sd = random_deletion(tokenized_sent, sdp_words,alpha_sr)
                    sent_sd=sent_sd.replace(self.el_sdp[0],self.el[0],1).replace(self.el_sdp[1],self.el[1],1)
                    print('Sentence with random deletion:',sent_sd)
                    senti.append(sent_sd)
                
                aug_sent_list.append(senti)
                f.write('\t'.join(senti))
                f.write('\n')

                

if __name__ == '__main__':
    #test
    '''
    for fi in range(1,11):
        file_path='./CLBERT/data/aimed/'+str(fi)+'/test.tsv'
        aug_file_path='./CLBERT/data/aimed/'+str(fi)+'/test_aug_sr.txt'
        task_name='ppi'
        eda=edaRE(file_path,aug_file_path,task_name)
        eda.augment(3,0,0)
    
    file_path='./CLBERT/data/chemprotms/test.tsv'
    aug_file_path='./CLBERT/data/chemprotms/test_aug_sr.txt'
    task_name='chemprot'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(3,0,0)

    file_path='./CLBERT/data/ddims/test.tsv'
    aug_file_path='./CLBERT/data/ddims/test_aug_sr.txt'
    task_name='ddi'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(3,0,0)

    
    file_path='./CLBERT/data/ds/mirgene/train.tsv'
    aug_file_path='./CLBERT/data/ds/mirgene/train_aug.txt'
    task_name='ds-mirgene'

    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(1,1,1)

    file_path='./CLBERT/data/mirgene/train.tsv'
    aug_file_path='./CLBERT/data/mirgene/train_aug_sr.txt'
    task_name='mirgene'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(3,0,0)

    file_path='./CLBERT/data/mirgene/train.tsv'
    aug_file_path='./CLBERT/data/mirgene/train_aug_rw.txt'
    task_name='mirgene'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(0,3,0)

    file_path='./CLBERT/data/mirgene/train.tsv'
    aug_file_path='./CLBERT/data/mirgene/train_aug_rd.txt'
    task_name='mirgene'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(0,0,3)

    file_path='./CLBERT/data/mirgene/train.tsv'
    aug_file_path='./CLBERT/data/mirgene/train_aug.txt'
    task_name='mirgene'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(1,1,1)
    

    file_path='./CLBERT/data/ds/chemprot/chemprot_ds_bert.tsv'
    aug_file_path='./CLBERT/data/ds/chemprot/train_aug.txt'
    task_name='ds-chemprot'

    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(1,1,1)
    
    file_path='./CLBERT/data/ds/ddi/ddi_ds_bert.tsv'
    aug_file_path='./CLBERT/data/ds/ddi/train_aug.txt'
    task_name='ds-ddi'

    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(1,1,1)
    '''
    
    file_path='./CLBERT/data/ds/ppi/train.tsv'
    aug_file_path='./CLBERT/data/ds/ppi/train_aug.txt'
    task_name='ds'

    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(1,1,1)

    '''
    file_path='./CLBERT/data/chemprot/train.tsv'
    aug_file_path='./CLBERT/data/chemprot/train_aug_sr.txt'
    task_name='chemprot'

    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(3,0,0)

    file_path='./CLBERT/data/aimed.txt'
    aug_file_path='./CLBERT/data/aimed_aug_sr.txt'
    task_name='ppi'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(3,0,0)

    file_path='./CLBERT/data/ddi/train.tsv'
    aug_file_path='./CLBERT/data/ddi/train_aug_sr.txt'
    task_name='ddi'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(3,0,0)
    
    #for random swap
    file_path='./CLBERT/data/chemprot/train.tsv'
    aug_file_path='./CLBERT/data/chemprot/train_aug_rw.txt'
    task_name='chemprot'

    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(0,3,0)

    file_path='./CLBERT/data/aimed.txt'
    aug_file_path='./CLBERT/data/aimed_aug_rw.txt'
    task_name='ppi'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(0,3,0)

    file_path='./CLBERT/data/ddi/train.tsv'
    aug_file_path='./CLBERT/data/ddi/train_aug_rw.txt'
    task_name='ddi'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(0,3,0)


    #for random deletion
    file_path='./CLBERT/data/chemprot/train.tsv'
    aug_file_path='./CLBERT/data/chemprot/train_aug_rd.txt'
    task_name='chemprot'

    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(0,0,3)

    file_path='./CLBERT/data/aimed.txt'
    aug_file_path='./CLBERT/data/aimed_aug_rd.txt'
    task_name='ppi'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(0,0,3)

    file_path='./CLBERT/data/ddi/train.tsv'
    aug_file_path='./CLBERT/data/ddi/train_aug_rd.txt'
    task_name='ddi'
    eda=edaRE(file_path,aug_file_path,task_name)
    eda.augment(0,0,3)
    '''