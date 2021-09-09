from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, EvalPrediction, \
    BertConfig
import torch
import argparse
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pipeline")
    parser.add_argument("--input", default="", help='train dataset')
    parser.add_argument("--pretrained_model", default="../output", help='pretrained model')

    args = parser.parse_args()

    config = json.load(open(os.path.join(args.pretrained_model, "config.json"), "r"))
    label_list = [config["id2label"][key] for key in sorted(config["id2label"].keys())]

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model, num_labels=len(label_list))

    config = BertConfig.from_pretrained(args.pretrained_model, num_labels=len(label_list))

    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(args.input)))
    inputs = tokenizer.encode(args.input, return_tensors="pt")

    outputs = model(inputs)[0]
    predictions = torch.argmax(outputs, dim=2)

    predict = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]

    print(predict)
    entities = []
    entity = ""
    for token, label in predict:
        if label.startswith('I-') or (label.startswith('B-') and not entity):
            if token.startswith("##"):
                entity += token.replace("##", "")
            else:
                entity += " " + token
        elif entity:
            entities.append(entity.strip())
            entity = ""
            if label.startswith('B-'):
                if token.startswith("##"):
                    entity += token.replace("##", "")
                else:
                    entity += " " + token
    print(entities)
