import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer

class TranslationDataset(Dataset):
    def __init__(self, en, hu, tokenizer, max_length = 128):
        self.en = en
        self.hu = hu
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.en)

    def __getitem__(self, idx):
        en = self.en[idx]
        hu = self.hu[idx]

        inputs = self.tokenizer(en, max_length = self.max_length, padding = 'max_length', truncation = True, return_tensors = 'pt')
        targets = self.tokenizer(hu, max_length = self.max_length, padding = 'max_length', truncation = True, return_tensors = 'pt')

        return inputs.input_ids.squeeze(), targets.input_ids.squeeze()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
english_texts = ["Hello world", "How are you?", "Good morning"] * 100  # Dummy data
hungarian_texts = ["Helló világ", "Hogy vagy?", "Jó reggelt"] * 100

dataset = TranslationDataset(english_texts, hungarian_texts, tokenizer)
train_size = int(0.8*len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
