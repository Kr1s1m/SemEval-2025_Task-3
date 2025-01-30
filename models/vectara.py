import json
from transformers import AutoModelForSequenceClassification

def read_jsonl(file_path):
    objects = []
    with open(file_path, 'r') as file:
        for line in file:
            obj = json.loads(line)
            objects.append(obj)
    return objects

train_samples = read_jsonl('data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl')

pairs = [(train_sample['model_input'], train_sample['model_output_text']) for train_sample in train_samples]

model = AutoModelForSequenceClassification.from_pretrained(
    'vectara/hallucination_evaluation_model', trust_remote_code=True)


predictions = model.predict(pairs)

print(predictions)