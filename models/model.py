from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, EvalPrediction, \
    AutoConfig
from seqeval.metrics import f1_score, precision_score, recall_score
import numpy as np
from typing import Dict, Tuple, List
from torch import nn
import other.utils as utils
import os


class Model:
    def __init__(self, configs):
        self.train_texts = configs.get("train_texts")
        self.train_labels = configs.get("train_labels")
        self.test_texts = configs.get("test_texts")
        self.test_labels = configs.get("test_labels")
        self.val_texts = configs.get("val_texts")
        self.val_labels = configs.get("val_labels")
        self.pretrained_model = configs.get("pretrained_model")
        self.output = configs.get("output")
        self.labels = configs.get("labels")
        self.num_train_epochs = configs.get("num_train_epochs")
        self.train_batch_size = configs.get("train_batch_size")
        self.eval_batch_size = configs.get("eval_batch_size")
        self.warmup_steps = configs.get("warmup_steps")
        self.weight_decay = configs.get("weight_decay")
        self.logging_steps = configs.get("logging_steps")
        self.do_train = configs.get("do_train")
        self.do_predict = configs.get("do_predict")

    def run_ner(self):
        # create encodings for tags
        unique_labels = set(tag for doc in self.labels for tag in doc)
        label2id = {label: i for i, label in enumerate(unique_labels)}
        id2label: Dict[int, str] = {i: label for i, label in enumerate(unique_labels)}
        num_labels = len(unique_labels)

        config = AutoConfig.from_pretrained(
            self.pretrained_model,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        model = AutoModelForTokenClassification.from_pretrained(self.pretrained_model, config=config)

        # Use a pretrained tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        def encode_tags(tags, encodings):
            labels = [[label2id[tag] for tag in doc] for doc in tags]
            encoded_labels = []
            for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
                # create an empty array of -100
                doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
                arr_offset = np.array(doc_offset)

                # set labels whose first offset position is 0 and the second is not 0
                doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
                encoded_labels.append(doc_enc_labels.tolist())

            return encoded_labels

        if self.do_train:
            training_args = TrainingArguments(
                output_dir=self.output,
                num_train_epochs=self.num_train_epochs,
                per_device_train_batch_size=self.train_batch_size,
                per_device_eval_batch_size=self.eval_batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir='./logs',
                logging_steps=self.logging_steps,
            )
            # tokenizer for train set
            train_encodings = tokenizer(self.train_texts, is_split_into_words=True, return_offsets_mapping=True,
                                        padding=True)
            # tokenizer for dev set
            val_encodings = tokenizer(self.val_texts, is_split_into_words=True, return_offsets_mapping=True,
                                      padding=True)

            train_labels = encode_tags(self.train_labels, train_encodings)
            val_labels = encode_tags(self.val_labels, val_encodings)

            # we don't want to pass this to the model
            train_encodings.pop("offset_mapping")
            val_encodings.pop("offset_mapping")

            self.train_texts = utils.Dataset(train_encodings, train_labels)
            self.val_texts = utils.Dataset(val_encodings, val_labels)

        if self.do_predict:
            training_args = TrainingArguments(output_dir=self.output)
            # tokenizer of test set
            test_encodings = tokenizer(self.test_texts, is_split_into_words=True, return_offsets_mapping=True,
                                       padding=True)
            test_labels = encode_tags(self.test_labels, test_encodings)
            test_encodings.pop("offset_mapping")
            self.test_texts = utils.Dataset(test_encodings, test_labels)

        def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
            preds = np.argmax(predictions, axis=2)

            batch_size, seq_len = preds.shape

            out_label_list = [[] for _ in range(batch_size)]
            preds_list = [[] for _ in range(batch_size)]

            for i in range(batch_size):
                for j in range(seq_len):
                    if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                        out_label_list[i].append(id2label[label_ids[i][j]])
                        preds_list[i].append(id2label[preds[i][j]])

            return preds_list, out_label_list

        def compute_metrics(p: EvalPrediction) -> Dict:
            preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)

            return {
                "precision": precision_score(out_label_list, preds_list),
                "recall": recall_score(out_label_list, preds_list),
                "f1": f1_score(out_label_list, preds_list),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_texts,
            eval_dataset=self.val_texts,
            compute_metrics=compute_metrics,
        )

        if self.do_train:

            trainer.train()

            # Evaluation
            results = {}
            if training_args.do_eval:

                result = trainer.evaluate()

                output_eval_file = os.path.join(self.output, "eval_results.txt")
                with open(output_eval_file, "w") as writer:
                    for key, value in result.items():
                        writer.write("%s = %s\n" % (key, value))

                results.update(result)
            # trainer.save_model()
            model_to_save = AutoModelForTokenClassification.from_pretrained(
                self.pretrained_model,
                config=config,
            )
            model_to_save.save_pretrained(self.output)
            tokenizer.save_pretrained(self.output)

        if self.do_predict:
            # predict
            predictions, label_ids, metrics = trainer.predict(self.test_texts)
            # preds_list, _ = align_predictions(predictions, label_ids)

            # Save predictions
            output_test_results_file = os.path.join(self.output, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    writer.write("%s = %s\n" % (key, value))
