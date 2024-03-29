import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import evaluate

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1_metric.compute(predictions=predictions, references=labels, average='macro')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_dir', default='Data/', type=str)
    parser.add_argument('--workspace', default='Workspaces/Test', type=str)
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()

    # fix all random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True

    filename = '{}.tsv'.format(args.dataset)
    df = pd.read_csv(
        os.path.join(args.data_dir, filename), 
        sep='\t'
    )

    label2id = {label:i for i, label in enumerate(df['label'].unique())}
    id2label = {i:label for label, i in label2id.items()}

    df['label'] = df['label'].map(label2id)

    # rename the column of df
    df = df.rename(columns={'context': 'text'})

    # drop the split column
    train_df = df[df['split'] == 'train'].drop(columns=['split']).reset_index(drop=True)
    val_df = df[df['split'] == 'val'].drop(columns=['split']).reset_index(drop=True)
    test_df = df[df['split'] == 'test'].drop(columns=['split']).reset_index(drop=True)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    modelname = 'xlnet-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, return_tensors="pt", padding=True)

    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_val = val_ds.map(preprocess_function, batched=True)

    f1_metric = evaluate.load("f1")

    training_args = TrainingArguments(
        output_dir=args.workspace,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=16,
        max_steps=4000,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        warmup_steps=500,
        save_steps=500,
        seed=args.seed,
        data_seed=args.seed,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # model = trainer.model
    
    model.cuda()
    model.eval()

    preds = []
    with torch.no_grad():
        for text in test_df['text']:
            tokenized_text = tokenizer(text, truncation=True, max_length=256, return_tensors="pt", padding=True)
            tokenized_text = tokenized_text.to('cuda')
            outputs = model(**tokenized_text)
            preds.append(outputs.logits.cpu().numpy().argmax())
    
    test_df['pred'] = preds
    metric = f1_score(test_df['label'], test_df['pred'], average='macro')

    print('Dataset: {}, seed: {}, F1 Score: {:.4f}'.format(args.dataset, args.seed, metric))