from torch.utils.data import DataLoader, Dataset

class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, source_lang, target_lang, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source_text = self.dataset[idx]['translation'][self.source_lang]
        target_text = self.dataset[idx]['translation'][self.target_lang]
        
        source_tokens = self.tokenizer(source_text, return_tensors="np", padding="max_length", truncation=True, max_length=self.max_length)
        target_tokens = self.tokenizer(target_text, return_tensors="np", padding="max_length", truncation=True, max_length=self.max_length)
        
        return {'src': source_tokens['input_ids'].squeeze(), 'tgt': target_tokens['input_ids'].squeeze()}


