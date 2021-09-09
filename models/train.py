import other.utils as utils
from sklearn.model_selection import train_test_split
import argparse
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NER")
    parser.add_argument("--train_data", default="../data/NERdata/train.tsv", help='train dataset')
    parser.add_argument("--test_data", default="../data/NERdata/test.tsv", help='Test dataset')
    parser.add_argument("--pretrained_model", default="dmis-lab/biobert-base-cased-v1.2", help='pretrained model')
    parser.add_argument("--output", default="../output", help='Output directory')
    parser.add_argument("--max_length", default=192)
    parser.add_argument("--num_train_epochs", default=30)
    parser.add_argument("--train_batch_size", default=32)
    parser.add_argument("--eval_batch_size", default=64)
    parser.add_argument("--warmup_steps", default=500)
    parser.add_argument("--weight_decay", default=0.01)
    parser.add_argument("--logging_steps", default=10)
    parser.add_argument("--do_train", default=True, action='store_true')
    parser.add_argument("--do_predict", default=False, action='store_true')

    args = parser.parse_args()

    # Read train dataset and set max length for each example
    texts_train_dev, labels_train_dev = utils.read_data(args.train_data, args.max_length)

    # Create a train/validation split
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts_train_dev, labels_train_dev, test_size=.2)

    config = {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "val_texts": val_texts,
        "val_labels": val_labels,
        "pretrained_model": args.pretrained_model,
        "output": args.output,
        "labels": labels_train_dev,
        "num_train_epochs": args.num_train_epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "do_predict": args.do_predict,
        "do_train": args.do_train
    }

    train_instance = Model(config)
    train_instance.run_ner()
