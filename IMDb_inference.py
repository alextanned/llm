from time import time
from vllm import LLM, SamplingParams
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import argparse
import os
from sklearn.model_selection import train_test_split


random.seed(6)

def fetch_reviews(path):
  data = []
#   path = 'aclImdb/train/pos/'

#       path = 'aclImdb/train/unsup/'

  files = [f for f in os.listdir(path)]
  for file in files:
    with open(path+file, "r", encoding='utf8') as f:
      data.append(f.read())
      
  return data


def data_prep(args):
    if args.extra:
        unlabeled = fetch_reviews('../../data/aclImdb/train/unsup/')
        return unlabeled, None, 2

    train_data = pd.read_pickle("../../data/IMDB_Finetune/train/train_data.pkl")
    val_data = pd.read_pickle("../../data/IMDB_Finetune/val/val_data.pkl")

    if args.split == 'train':
        X_test, y_test = train_data['review'], train_data['label']
    elif args.split == 'val':
        X_test, y_test = val_data['review'], val_data['label']

    num_classes = 2

    return X_test, y_test, num_classes



def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some parameters for model training.")

    parser.add_argument(
        '--save-path',
        type=str,
        required=False,
        default="./",
        help='The file path where output file will be saved.'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='The name of the model to be used.'
    )

    parser.add_argument(
        '--split',
        type=str,
        required=False,
        default='val',
    )  

    parser.add_argument(
        '--binary',
        action='store_true',
        default=False,
        help='Specify whether to perform binary classification.'
    )

    parser.add_argument(
        '--ignore-other',
        action='store_true',
        default=False,
        help='Ignore ill-formed class \'other\''
    )

    parser.add_argument(
        '--score-optimized',
        action='store_true',
        default=False,
        help='Optimize threshold for maximum accuracy'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='debug'
    )

    parser.add_argument(
        '--extra',
        action='store_true',
        default=False,
        help='extra unlabeled data'
    )

    args = parser.parse_args()
    return args


def find_best_thresholds(numerical_outputs, y_true, num_classes):
    best_accuracy = 0
    best_thresholds = (0, 100) 

    for t1 in np.linspace(0, 100, 101, dtype=np.int8):
        for t2 in np.linspace(t1, 100, 101-t1, dtype=np.int8):
            bin_edges = np.array([0, t1, t2, 100])
            binned_numbers = np.digitize(numerical_outputs, bins=bin_edges, right=False) - 1
            y_pred = np.clip(binned_numbers, 0, num_classes - 1) # for the number 100 to be put in last class
            y_pred+=1
            accuracy = accuracy_score(y_true, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thresholds = (t1, t2)
                print(best_thresholds)
    
    return best_thresholds, best_accuracy

def inference(args):
    
    X_test, y_true, num_classes = data_prep(args)
    if args.debug:
        X_test = X_test[:100]
        y_true = y_true[:100]
        
    # Create an LLM.
    # regular_llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.4)
    llm = LLM(
        model=args.model, #meta-llama/Meta-Llama-3-8B-Instruct
        enable_prefix_caching=True,
        max_model_len=8192,
        disable_sliding_window=True,
        # gpu_memory_utilization=1.0,
        max_logprobs=100,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=128,
            logprobs=10,
            # logit_bias={43324:10, 31587:10}
            # stop_token_ids=[regular_tokenizer.eos_token_id, regular_tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # KEYPOINT HERE
        )


    prefix = (
        # "Please answer with 'positive' or 'negative' only! "
        "Decide if the following movie review is positive or negative: "
        "{0} "
        "If the movie review is positive please answer 'positive', "
        "if the movie review is negative please answer 'negative'. "
        "Make your decision based on the whole text."
    )
 
    # prefix  = (
    #     "Rate the sentiment of the following movie review as positive or negative: " 
    #     "{{{0}}} "
    #     "Provide a number 0 or 1, where: "
    #     "0 represents the movie review is negative, " 
    #     "and 1 represents the moview review is positive. "
    #     "Do not provide any explanations, only respond with the number."
    # )
    # prompts = test_data['statement'].sample(frac = 1)

    if args.ignore_other:
        # y_true =  test_data[test_data['gold_label'] != '-']['output']
        # prompt1 = test_data[test_data['gold_label'] != '-']['sentence1']
        # prompt2 = test_data[test_data['gold_label'] != '-']['sentence2']
        pass
    else:

        prompts = X_test


    generating_prompts = tokenizer.apply_chat_template([
        [   {"role": "system", "content": "Please answer with 'positive' or 'negative' only! "},
            {"role": "user", "content": prefix.format(prompt)}] for prompt in prompts
    ], tokenize=False)

    outputs = llm.generate(generating_prompts,     
        sampling_params   )

    numerical_outputs = []
    for i, output in enumerate(outputs):
        # print(repr(output.outputs[0].text))
        # print()
        try:

            rating = output.outputs[0].text.split("\n")[-1].lower()

            # if rating == "negative":
            #     number = 0
            # elif rating == "positive":
            #     number = 1
            if rating.startswith("negative"):
                number = 0
            elif rating.startswith("positive"):
                number = 1
            else:
                raise ValueError
            # number = int(output.outputs[0].text.split("\n")[-1])
        except ValueError:
            print(output.outputs[0].text)
            # If conversion fails, replace with a random integer between 0 and 100
            number = random.randint(0, 1)
        numerical_outputs.append(number)

    
    if args.extra:
        files = [f for f in os.listdir('../../data/aclImdb/train/unsup/')]
        labeled = pd.DataFrame({'review': X_test, 'filename':files, 'label': numerical_outputs})
        labeled.to_pickle("../../data/IMDB_Finetune/extra/extra_llama3.1.pkl")
        exit()


    
    if args.score_optimized:
        thresholds, acc = find_best_thresholds(numerical_outputs, y_true, num_classes)
        print(thresholds)
    else:
        # bin_edges = np.linspace(0, 100, num_classes + 1)
        # binned_numbers = np.digitize(numerical_outputs, bins=bin_edges, right=False) - 1
        # y_pred = np.clip(binned_numbers, 0, num_classes - 1) # for the number 100 to be put in last class
        # print(y_pred)
        y_pred = numerical_outputs     
        acc = accuracy_score(y_true, y_pred)


    


    print(acc)

    # test_data['pred'] = numerical_outputs
    # test_data[['our rating', 'text', 'output', 'pred']].to_csv(os.path.join(args.save_path, "pred.csv"))
    with open(os.path.join(args.save_path, "results.txt"), "w") as f:
        f.write(str(acc))
        f.writelines([o.outputs[0].text.split("\n")[-1] for o in outputs])
    
    return outputs

if __name__ == "__main__":
    args = parse_arguments()

    output = inference(args)

