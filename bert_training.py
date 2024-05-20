import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_for_bert(texts, labels, tokenizer, max_length=64):  # Pass tokenizer as a parameter
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length)
    return Dataset(encodings, labels.tolist())

def train_bert(train_dataset, test_dataset, model_name='distilbert-base-uncased', num_labels=3, num_epochs=1, model_path='./saved_model'):  # Reduced epochs to 1
    # Check if the model path exists
    if os.path.exists(model_path):
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    else:
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Check if CUDA is available and set FP16 training accordingly
    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,  # Adjusted batch size for faster training
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        fp16=use_cuda,  # Enable mixed precision training only if CUDA is available
        gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
        save_steps=10_000,  # Save every 10,000 steps
        save_total_limit=2,  # Only keep the last two versions
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    if not os.path.exists(model_path):
        trainer.train()
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    return trainer

def evaluate_bert(trainer):
    return trainer.evaluate()

def main():
    # Load and preprocess the data
    filepath = 'labeled_data.csv'
    preprocessed_data = preprocess_data(filepath, 'tweet')

    # Split the data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        preprocessed_data['processed_text'], preprocessed_data['class'], test_size=0.2, random_state=42)

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  # Use DistilBERT for faster training

    # Preprocess for BERT
    train_dataset = preprocess_for_bert(train_texts, train_labels, tokenizer)
    test_dataset = preprocess_for_bert(test_texts, test_labels, tokenizer)

    # Train BERT model
    trainer = train_bert(train_dataset, test_dataset)

    # Evaluate BERT model
    evaluation_results = evaluate_bert(trainer)
    print("BERT Evaluation Results:", evaluation_results)

if __name__ == '__main__':
    main()
