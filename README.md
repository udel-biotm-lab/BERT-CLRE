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
We implement our project with Tensorflow 1.15 and we utilize the pre-trained BioBERT/PubMedBERT model as our initial model for contrastive pre-training. We can use the following code for the contrastive pre-training:
$TASK_NAME='aimed' or 'ddi13' or 'chemprot';\
$BERT_DIR is the path where we store the pre-trained BERT model;\
$RE_DIR is the path where we have the contrastive learning dataset;\
$OUTPUT_DIR is the path where we can store the contrastively pre-trained BERT model;
```
TASK_NAME="task_name"
BERT_DIR="./biobert_v1.1_pubmed"

RE_DIR="./REdata/contrastive_pre-training_dataset/"
OUTPUT_DIR="./REoutput/model_output"


for i in 2 4 6 8 10
do

	python run_re_cp.py --task_name=$TASK_NAME --do_train=true --do_eval=false --do_predict=false --vocab_file=$BERT_DIR/vocab.txt --bert_config_file=$BERT_DIR/bert_config.json --init_checkpoint=$BERT_DIR/model.ckpt-1000000 --max_seq_length=128 --train_batch_size=256 --learning_rate=2e-5 --num_train_epochs=${i} --do_lower_case=false --data_dir=${RE_DIR} --output_dir=${OUTPUT_DIR} --model_name="cl_pretraining"

done

```
The datasets for contrastive pre-training are available at:
1. Contrastive pre-training dataset for [PPI](https://drive.google.com/drive/folders/1fANJLDWxmYgbrAQUqgAlKl9RzcofCHob?usp=sharing)
2. Contrastive pre-training dataset for [DDI](https://drive.google.com/drive/folders/1IYWJJRDusxUwYfn37jC4k1m3WSyu8FOI?usp=sharing)
3. Contrastive pre-training dataset for [ChemProt](https://drive.google.com/drive/folders/1l_mDKPOMMaiXkYeuynq3aeVj0902Sbyw?usp=sharing)

### 4. Fine-tuning of BERT model:
After the pre-training, we then can fine-tune the BERT model on the evaluation sets of PPI, DDI and ChemProt:\
```
TASK_NAME="task_name"
BERT_DIR="./REoutput/contrastive_pre_trained_model"

RE_DIR="./REdata/aimed/"

OUTPUT_DIR="./REoutput/model_output_folder"


for s in 2 4 6 8 10
do
	for i in {1..10}
	do
		python run_re.py --task_name=$TASK_NAME --do_train=true --do_eval=false --do_predict=true --vocab_file=$BERT_DIR/vocab.txt --bert_config_file=$BERT_DIR/bert_config.json --init_checkpoint=$BERT_DIR/model.ckpt-1528 --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=${s} --do_lower_case=false --data_dir=${RE_DIR}${i} --output_dir=${OUTPUT_DIR}${i}


	done

	python ./biocodes/re_eval.py --output_path=${OUTPUT_DIR} --answer_path=${RE_DIR} --fold_number=10 --step=${s} --task_name="aimed_constrastive_learning"


done
```

For the PPI task, we are using 10 fold cross-validation (as shown above), but for DDI and ChemProt, we do not have to do this.
