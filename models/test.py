import other.utils as utils
from sklearn.model_selection import train_test_split
import argparse
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NER")
    parser.add_argument("--train_data", default="../data/NERdata/train.tsv", help='train dataset')
    parser.add_argument("--test_data", default="../data/NERdata/test.tsv", help='Test dataset')
    parser.add_argument("--pretrained_model", default="../output", help='pretrained model')
    parser.add_argument("--output", default="../output", help='Output directory')
    parser.add_argument("--max_length", default=128)
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument("--do_predict", default=True, action='store_true')
    args = parser.parse_args()

    # Read train dataset and set max length for each example
    test_texts, test_labels = utils.read_data(args.test_data, args.max_length)

    config = {
        "test_texts": test_texts,
        "test_labels": test_labels,
        "pretrained_model": args.pretrained_model,
        "output": args.output,
        "labels": test_labels,
        "do_predict": args.do_predict,
        "do_train": args.do_train
    }

    predict_instance = Model(config)
    predict_instance.run_ner()
