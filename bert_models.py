from datasets import load_dataset
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import pandas as pd
import sklearn
import argparse


parser = argparse.ArgumentParser(description='Run BERT models for HS detection.')
parser.add_argument("--model", help="The model to use (bert/hatebert/fbert)", default="bert")
# parser.add_argument("--dataset", help="The dataset to use (hateval)", default=hateval)

args = parser.parse_args()

def load_hatexplain_data():

    def is_hatespeech(ls):
        '''
        The hatexplain data features three labels by different annotators, this
        method determines whether to label the row as hate or not. Only messages
        that receive a majority label of hatespeech (0) are considered as such.
        '''
        if sum((x == 0 for x in ls)) > len(ls)//2:
            return 1
        else:
            return 0

    print("loading data...")
    #load datasets
    train_data = load_dataset("hatexplain", split="train")
    test_data = load_dataset("hatexplain", split="test")
    #there is also a validation dataset, could load later

    print("processing data...")
    #convert to pandas
    train_data = train_data.to_pandas()
    test_data = test_data.to_pandas()

    #extract label sets
    train_label_sets= train_data["annotators"].map(lambda x: x["label"])
    train_labels = train_label_sets.map(is_hatespeech)

    test_label_sets= test_data["annotators"].map(lambda x: x["label"])
    test_labels = test_label_sets.map(is_hatespeech)

    #add label column to dataframe
    train_data["labels"] = train_labels
    test_data["labels"] = test_labels

    #detokenize posts
    train_data["text"] = train_data["post_tokens"].map(" ".join)
    test_data["text"] = test_data["post_tokens"].map(" ".join)

    return train_data, test_data


train_data, test_data = load_hatexplain_data()

#setup logger
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

print("loading model...")
# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=1)

# Create a ClassificationModel depending on provided model
if args.model == "bert":
    print("loading bert...")
    model = ClassificationModel(
        "bert", "bert-base-uncased", args=model_args, use_cuda=False
    )
elif args.model == "hatebert":
    print("loading hatebert...")
    model = ClassificationModel(
        "bert", "GroNLP/hateBERT", args=model_args, use_cuda=False,
        tokenizer_type="bert", tokenizer_name="GroNLP/hateBERT"
    )
elif args.model == "fbert":
    print("loading fbert...")
    model = ClassificationModel(
        "bert", "diptanu/fBERT", args=model_args, use_cuda=False,
        tokenizer_type="bert", tokenizer_name="diptanu/fBERT"
    )
else:
    raise ValueError("Provided model invalid. Specify bert, hatebert or fbert.")

print("training model...")
model.train_model(train_data)

print("evaluating model...")
result, model_outputs, wrong_predictions = model.eval_model(test_data)

print(result)
