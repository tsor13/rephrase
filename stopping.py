import torch
from transformers import StoppingCriteria, AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).eval()

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


stop_words = ['}', ' }', '\n']
stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
stop_criteria = KeywordsStoppingCriteria(stop_ids)


inputs = tokenizer.encode('some text: {', add_special_tokens=False, return_tensors='pt')

output = model.generate(
    inputs,
    do_sample=True,
    stopping_criteria=StoppingCriteriaList([stop_criteria]),

)
print(tokenizer.decode(*output))