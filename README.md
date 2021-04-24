### 1. Evaluation datasets for the PPI, DDI and ChemProt tasks: 
1. Dataset for [PPI](https://drive.google.com/file/d/1dn2yDKj7-3SsyKQ5Zm_5sTlLxTCfqQpy/view?usp=sharing)
2. Dataset for [DDI](https://drive.google.com/file/d/1EEtN1LMI-W4iqtsXVfc64v5PsoAEmJad/view?usp=sharing)
3. Dataset for [ChemProt](https://drive.google.com/file/d/1XSieVU673Ey52xSV16pZ7a_8fqBJFd6k/view?usp=sharing)

### 2. Augmented evaluation datasets for the PPI, DDI and ChemProt tasks:
1. Dataset for [PPI](https://drive.google.com/file/d/1GUJdJo-ihl2StJMNNyqrPXvNDyKolaTI/view?usp=sharing)
2. Dataset for [DDI](https://drive.google.com/file/d/1lTo_yk9J0sJuBy-lXiGDx9SqNZ2UjrE8/view?usp=sharing)
3. Dataset for [ChemProt](https://drive.google.com/file/d/1scoGLZAoyM9j9ebvVW9BA1TsjvNO0Xrz/view?usp=sharing)

In those augmented datasets, we also include the original data instances, a.k.a., the format of the file is: \
line N-1: original data\
Line N: augmented data.

### 3. Contrastive pre-training procedure: 


```


```


After the pre-training, we then can fine-tune the BERT model on the evaluation sets of PPI, DDI and ChemProt:\
$TASK_NAME='aimed' or 'ddi13' or 'chemprot';\
$BERT_DIR is the path where we store the pre-trained BERT model using sub-domain data;\
$RE_DIR is the path where we have the evaluation set;\
$OUTPUT_DIR is the path where we can store the fine-tuned BERT model;
```

```
### 4. Fine-tuning of BERT model:

```

```

For the PPI task, we are using 10 fold cross-validation, but for DDI and ChemProt, we do not have to do this.
