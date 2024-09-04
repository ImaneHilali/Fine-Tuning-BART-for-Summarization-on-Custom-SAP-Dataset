import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from rouge_score import rouge_scorer
import numpy as np

model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

OPTIMAL_MAX_LENGTH = 512

fine_tuned_model_path = "C:\\Fine-tunning\\fine_tuned_bart_sap_model.pth"

try:
    model.load_state_dict(torch.load(fine_tuned_model_path))
    print("Fine-tuned model loaded successfully!")
except FileNotFoundError:
    print("Fine-tuned model not found!")


class SAPDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


df = pd.read_csv("C:\\Users\\emy7u\\Downloads\\tldr-challenge-dataset\\fine-tunning.csv")

actual_dataset = [(row["text"], row["summary"]) for index, row in df.iterrows()]

dataset = SAPDataset(actual_dataset)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    for batch in dataloader:
        input_text, target_text = batch
        input_ids = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=1024,
                              add_special_tokens=True)['input_ids']
        target_ids = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=1024,
                               add_special_tokens=True)['input_ids']

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss
        loss.backward()

        # Gradient clipping is a technique to prevent the exploding gradient problem, where gradients during backpropagation become too large, leading to numerical instability and poor training performance. By clipping, gradients exceeding a certain threshold are scaled down to keep them within a manageable range. This ensures stable and efficient training of deep neural networks.
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        print(f"Loss: {loss.item()}")


torch.save(model.state_dict(), fine_tuned_model_path)
print("Fine-tuned model saved successfully!")

print(f"Suggested max_length based on 95th percentile: {max_length}")

def generate_summary(text):

    inputs = tokenizer(text, return_tensors='pt', truncation=True, add_special_tokens=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=300, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summary



# Usage Example 
sample_document = "text sample..."

summary = generate_summary(sample_document)
print("Generated Summary:", summary)