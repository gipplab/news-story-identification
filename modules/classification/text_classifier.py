from pandas import DataFrame
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_text = 'I will watch Memento tonight'
bert_input = tokenizer(example_text,padding='max_length', max_length = 10, 
                       truncation=True, return_tensors="pt")


print(bert_input['input_ids'])
print(bert_input['token_type_ids'])
print(bert_input['attention_mask'])

class TextClassifier():
    def __init__(self, data: DataFrame, label: str) -> None:
        super().__init__()
        self.data = data
        self.label = label

    def __str__(self) -> str:
        pass

    def k_fold(self, k: int):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass