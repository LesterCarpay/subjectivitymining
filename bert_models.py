from datasets import load_dataset
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import pandas as pd
import argparse
import torch
import os



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

    #load datasets
    train_data = load_dataset("hatexplain", split="train")
    test_data = load_dataset("hatexplain", split="test")
    #there is also a validation dataset, could load later

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

def load_olid_data():
    train_path = os.path.join("data", "OLID", "trainData.csv")
    test_path = os.path.join("data", "OLID", "testData.csv")

    train_data = pd.read_csv(train_path, delimiter="\t")
    test_data = pd.read_csv(test_path, delimiter="\t")

    train_data.rename({"Text": "text"}, axis=1, inplace=True)
    test_data.rename({"Text": "text"}, axis=1, inplace=True)

    train_data["labels"] = train_data["Label"].map(lambda x: int(x == "OFF"))
    test_data["labels"] = test_data["Label"].map(lambda x: int(x == "OFF"))

    return train_data, test_data

def load_davidson_data():
  #load datasets
  data = load_dataset("hate_speech_offensive", split="train")
  #1/3 = test
  train_data=data[0:16522]
  test_data=data[16522:24783]

  print("processing data...")
  #convert to pandas
  train_data = pd.DataFrame.from_dict(train_data)
  test_data = pd.DataFrame.from_dict(test_data)

  train_data["labels"] = train_data["hate_speech_count"]
  test_data["labels"] = test_data["hate_speech_count"]

  train_data["text"] = train_data["tweet"]
  test_data["text"] = test_data["tweet"]
  train_data=train_data[["labels","text"]]
  test_data=test_data[["labels","text"]]

  return train_data, test_data

def load_data(dataset):
    print("loading", dataset, "data...")
    if dataset == "hatexplain":
        return load_hatexplain_data()
    if dataset == "olid":
        return load_olid_data()
    if dataset == "davidson":
        return load_davidson_data()
    raise ValueError("Provided model invalid, choose hatexplain, olid or davidson.")

def create_model(model_name, use_cuda= False):
    #setup logger
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)


    print("loading model...")
    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=3, train_batch_size=1)

    # Create a ClassificationModel depending on provided model
    if model_name == "bert":
        print("loading bert...")
        model = ClassificationModel(
            "bert", "bert-base-uncased", args=model_args, use_cuda=use_cuda
        )
    elif model_name == "hatebert":
        print("loading hatebert...")
        model = ClassificationModel(
            "bert", "GroNLP/hateBERT", args=model_args, use_cuda=use_cuda,
            tokenizer_type="bert", tokenizer_name="GroNLP/hateBERT"
        )
    elif model_name == "fbert":
        print("loading fbert...")
        model = ClassificationModel(
            "bert", "diptanu/fBERT", args=model_args, use_cuda=use_cuda,
            tokenizer_type="bert", tokenizer_name="diptanu/fBERT"
        )
    else:
        raise ValueError("Provided model invalid. Specify bert, hatebert or fbert.")
    return model


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Run BERT models for HS detection.')
    parser.add_argument("--model", help="The model to use (bert/hatebert/fbert)", default="bert")
    parser.add_argument("--dataset", help="The dataset to use (hatexplain/olid/davidson)", default="hatexplain")

    args = parser.parse_args()

    train_data, test_data = load_data(args.dataset)

    model = create_model(args.model, use_cuda= True)

    torch.cuda.empty_cache()

    print("training model...")
    model.train_model(train_data)

    print("evaluating model...")
    result, model_outputs, wrong_predictions = model.eval_model(test_data)

    precision = result["tp"]/(result["tp"] + result["fp"])
    recall = result["tp"]/(result["tp"] + result["fn"])
    f1 = 2*precision*recall/(precision + recall)

    print("Precision:", precision)
    print("Recall:", recall)
    print("f1:", f1)
