import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from googleapiclient import discovery
import json
from fuzzywuzzy import fuzz

model_size = 'large'
model_name = 't5-{}'.format(model_size)
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def get_entailment_str(text1, text2):
    return 'mnli premise: {} hypothesis: {}'.format(text1, text2)

answer_dict = {'neutral': 7163, 'entailment': 3, 'contradiction': 27252}

def get_entailment_score(text1, text2):
    sentence = get_entailment_str(text1, text2)
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    # generate the output
    output = model.generate(input_ids)
    # print(tokenizer.decode(output[0]))
    # print(output[0])
    decoder_input_ids = output[:,0:1]

    # get the logits
    logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)[0]
    # extract the logits for the answer
    answer_logits = {k: np.exp(logits[0,0,v].item()) for k, v in zip(answer_dict.keys(), answer_dict.values())}
    # normalize the logits
    sum_logits = sum(answer_logits.values())
    answer_logits = {k: v/sum_logits for k, v in answer_logits.items()}
    # 1 for entailment, 0 for contradiction, -1 for neutral
    score = answer_logits['entailment'] - answer_logits['contradiction']
    return score

# API_KEY = 'AIzaSyAkk4vOo8VDbCm5r4kHfIKgPs4ZbpI-D7o'
API_KEY = 'AIzaSyDo6-i-vdA9V-qoV8kEbIGsWwTrwJdMN4g'
from googleapiclient import discovery
def get_toxicity_score(text):

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': { 'text': text},
        'requestedAttributes': {'TOXICITY': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    # print(json.dumps(response, indent=2))
    toxicity = response['attributeScores']['TOXICITY']['summaryScore']['value']
    return toxicity

def get_levenshtein(text1, text2):
    return fuzz.ratio(text1, text2) / 100

def get_reward(text1, text2):
    if len(text1) == 0 or len(text2) == 0:
        return -2
    # entailment - toxicity of text2
    entailment_score = (get_entailment_score(text1, text2) + get_entailment_score(text2, text1)) / 2
    levenshtein_score = get_levenshtein(text1, text2)
    toxicity_score = get_toxicity_score(text2)
    # print('entailment: {}, toxicity: {}'.format(entailment_score, toxicity_score))
    total_score = (entailment_score + levenshtein_score)/2 - toxicity_score
    print('entailment: {}, leveinstein: {}, toxicity: {}, total: {}'.format(entailment_score, levenshtein_score, toxicity_score, total_score))
    return total_score