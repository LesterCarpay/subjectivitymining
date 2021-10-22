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
  print("loading data...")
  #load datasets
  data = load_dataset("hate_speech_offensive", split="train")
  data = pd.DataFrame.from_dict(data)
  
  data["labels"]= data["class"]
  data["labels"][data["labels"] > 0] = -1
  data["labels"][data["labels"] ==0 ] = 1
  data["labels"][data["labels"] < 0] = 0

  data["text"] = data["tweet"]

  print("processing data...")

  data=data[["labels","text"]]

  x_train, x_test, y_train, y_test = train_test_split(data["text"],data["labels"],test_size=0.33,stratify=data["labels"])
  train_list = list(zip(x_train,y_train))
  test_list = list(zip(x_test,y_test))
  train_data = pd.DataFrame(train_list,columns = ["text","labels"])
  test_data = pd.DataFrame(test_list,columns = ["text","labels"])
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

def test_on_all_datasets(model):
    def print_results(name, result, wrong):
        print("-"*50)
        print(name)
        print("-"*50)
        precision = result["tp"]/(result["tp"] + result["fp"])
        recall = result["tp"]/(result["tp"] + result["fn"])
        f1 = 2*precision*recall/(precision + recall)

        print("Precision:", precision)
        print("Recall:", recall)
        print("f1:", f1)
        print("Examples of wrong predictions:")
        print(wrong[:10])

    print("evaluating model...")
    # test on hatexplain
    _, test_data = load_hatexplain_data()
    result_h, _, wrong_h = model.eval_model(test_data)


    # test on olid
    _, test_data = load_olid_data()
    result_o, _, wrong_o = model.eval_model(test_data)

    # test on davidson
    _, test_data = load_davidson_data()
    result_d, _, wrong_d = model.eval_model(test_data)

    #print results
    print_results("HATEXPLAIN", result_h, wrong_h)
    print_results("OLID", result_o, wrong_o)
    print_results("DAVIDSON", result_d, wrong_d)


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

    test_on_all_datasets(model)
