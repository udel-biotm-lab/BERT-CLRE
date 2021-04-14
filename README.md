### 1. Evaluation datasets for the PPI, DDI and ChemProt tasks: 
[PPI](https://drive.google.com/file/d/1dn2yDKj7-3SsyKQ5Zm_5sTlLxTCfqQpy/view?usp=sharing)\
[DDI](https://drive.google.com/file/d/1EEtN1LMI-W4iqtsXVfc64v5PsoAEmJad/view?usp=sharing)\
[ChemProt](https://drive.google.com/file/d/1XSieVU673Ey52xSV16pZ7a_8fqBJFd6k/view?usp=sharing)

### 2. Augmented evaluation datasets for the PPI, DDI and ChemProt tasks:

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
