from time import time
from vllm import LLM, SamplingParams
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score
import argparse
import os

random.seed(6)


def data_prep(args):
    test_data = pd.read_table('../../data/liar_dataset/test.tsv', sep='\\t',engine='python', header=None)
    test_data.fillna("", inplace=True)
    test_data.columns = [
            'id',                # Column 1: the ID of the statement ([ID].json).
            'label',             # Column 2: the label.
            'statement',         # Column 3: the statement.
            'subjects',          # Column 4: the subject(s).
            'speaker',           # Column 5: the speaker.
            'speaker_job_title', # Column 6: the speaker's job title.
            'state_info',        # Column 7: the state info.
            'party_affiliation', # Column 8: the party affiliation.
            
            # Column 9-13: the total credit history count, including the current statement.
            'count_1', # barely true counts.
            'count_2', # false counts.
            'count_3', # half true counts.
            'count_4', # mostly true counts.
            'count_5', # pants on fire counts.
            
            'context' # Column 14: the context (venue / location of the speech or statement).
        ]

    if args.binary:
        y_label_dict = {"pants-fire" : 0, "false" : 0, "barely-true" : 0, "half-true" : 1, "mostly-true" : 1, "true" : 1}
        num_classes = 2
    else:
        y_label_dict = {"pants-fire" : 0, "false" : 1, "barely-true" : 2, "half-true" : 3, "mostly-true" : 4, "true" : 5}
        num_classes = 6
    test_data['output'] = test_data['label'].apply(lambda x: y_label_dict[x])
    
    return test_data, num_classes



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
        '--binary',
        action='store_true',
        default=False,
        help='Specify whether to perform binary classification. Default is True. Set to False for multiclass classification.'
    )

    parser.add_argument(
        '--score-optimized',
        action='store_true',
        default=False,
        help='Optimize threshold for maximum accuracy'
    )

    parser.add_argument(
        '--few-shot',
        action='store_true',
        default=False,
    )

    args = parser.parse_args()
    return args

def find_best_thresholds(numerical_outputs, y_true, num_classes):
    best_accuracy = 0
    best_threshold = 0

    for t1 in np.linspace(0, 100, 101, dtype=np.int8):
        
        bin_edges = np.array([0, t1, 100])
        binned_numbers = np.digitize(numerical_outputs, bins=bin_edges, right=False) - 1
        y_pred = np.clip(binned_numbers, 0, num_classes - 1) # for the number 100 to be put in last class
        
        accuracy = accuracy_score(y_true, y_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t1
            print(best_threshold)
    
    return best_threshold, best_accuracy


def inference(args):
    test_data, num_classes = data_prep(args)
        
    # Create an LLM.
    # regular_llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.4)
    llm = LLM(
        model=args.model, #meta-llama/Meta-Llama-3-8B-Instruct
        enable_prefix_caching=True,
        max_model_len=256,
        disable_sliding_window=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=128,
            # stop_token_ids=[regular_tokenizer.eos_token_id, regular_tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # KEYPOINT HERE
        )

    # prepare data
    # Common prefix.
    prefix = ("Rate the truthfulness of the following statement: "
        "{{{0}}} Provide a score from 0 to 100, where 0 represents "
        "definitively false and 100 represents definitively true. "
        "Do not provide any explanations, only respond with the numerical score."
    )


    # prefix = ("Rate the truthfulness of the following statement: "
    #     "{{{0}}} Provide a number 0 or 1, where 0 represents false "
    #     "and 1 represents true. "
    #     "Do not provide any explanations, only respond with the number."
    #    )

    # prompts = test_data['statement'].sample(frac = 1)
    prompts = test_data['statement']
    

    generating_prompts = tokenizer.apply_chat_template([
        [{"role": "user", "content": prefix.format(prompt)}] for prompt in prompts
    ], tokenize=False)


    outputs = llm.generate(generating_prompts,     
        sampling_params   )


    numerical_outputs = []
    for i, output in enumerate(outputs):
        try:
            number = int(output.outputs[0].text.split("\n")[-1])
        except ValueError:
            # If conversion fails, replace with a random integer between 0 and 100
            number = random.randint(0, 100)
        numerical_outputs.append(number)

    y_true =  test_data['output']

    if args.score_optimized:
        threshold, acc = find_best_thresholds(numerical_outputs, y_true, num_classes)
        print(threshold)
    else:
        bin_edges = np.linspace(0, 100, num_classes + 1)
        binned_numbers = np.digitize(numerical_outputs, bins=bin_edges, right=False) - 1
        y_pred = np.clip(binned_numbers, 0, num_classes - 1) # for the number 100 to be put in last class
        acc= accuracy_score(y_true, y_pred)
    
    print(acc)

    test_data['pred'] = numerical_outputs
    test_data[['label', 'statement', 'output', 'pred']].to_csv(os.path.join(args.save_path, "pred.csv"))
    with open(os.path.join(args.save_path, "results.txt"), "w") as f:
        f.write(str(acc))

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    inference(args)
    # print(f"CSV will be saved at: {args.save_csv}")
    # print(f"Model name: {args.model_name}")
    # print(f"Classification type: {args.classification_type}")
