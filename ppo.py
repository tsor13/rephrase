import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from stopwords import KeywordsStoppingCriteria
from reward import get_reward

tokenizer = AutoTokenizer.from_pretrained('gpt2')

stop_words = ['}', ' }', '\n']
stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
stop_criteria = KeywordsStoppingCriteria(stop_ids)

def get_original_and_rephrasing(text):
    original = text.split('{')[1].split('}')[0]
    rephrasing = text.split('{')[2].split('}')[0]
    return original, rephrasing

def rollout(model, tokenizer, prompt, n=1, temp=1, top_k=50, top_p=1):
    '''
    Given the model, tokenize the prompt and generate n times.
    '''
    output_ids = []
    texts = []
    rewards = []
    rephrasings = []
    loss_masks = []
    # tokenize
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
    for _ in range(n):
        # generate
        # TODO - add bad words ids?
        with torch.no_grad():
            if temp > 0:
                output = model.generate(input_ids, max_length=200, temperature=temp,
                    top_k=top_k, top_p=top_p, do_sample=True, min_length=len(input_ids[0])+2,
                    stopping_criteria=StoppingCriteriaList([stop_criteria]))
            else:
                output = model.generate(input_ids, do_sample=False, max_length=200,
                    min_length=len(input_ids[0])+2,
                    stopping_criteria=StoppingCriteriaList([stop_criteria]))
        output_ids.append(output)
        # loss mask are the added tokens
        loss_mask = torch.zeros_like(output)
        loss_mask[:, len(input_ids[0])+1:] = 1
        loss_masks.append(loss_mask)
        # decode
        text = tokenizer.decode(*output, skip_special_tokens=True)
        texts.append(text)
        # reward
        original, rephrasing = get_original_and_rephrasing(text)
        print(rephrasing)
        reward = get_reward(original, rephrasing)
        rewards.append(reward)
        # append rephrasing
        rephrasings.append(rephrasing)
    
    return {
        'output_ids': output_ids,
        'texts': texts,
        'rewards': rewards,
        'rephrasings': rephrasings,
        'loss_masks': loss_masks
    }

def templatize(text):
    return 'Please rephrase the following text to be less toxic: {' + text + '}\nRephrasing: {'

def train(model, original_model, tokenizer, optimizer, texts, epochs=10, batch_size=4):
    average_rewards = []
    for epoch in range(epochs):
        for text in texts:
            prompt = templatize(text)
            rollout_data = rollout(model, tokenizer, prompt, n=batch_size)
            baseline_reward = rollout(model, tokenizer, prompt, n=1, temp=0)['rewards'][0]
            normalized_rewards = np.array(rollout_data['rewards']) - baseline_reward
            rollout_data['normalized_rewards'] = normalized_rewards

            print(np.mean(rollout_data['rewards']))
            average_rewards.append(np.mean(rollout_data['rewards']))

            # A is normalized reward
            A = torch.tensor(normalized_rewards, dtype=torch.float)
            # run through model and original model
            output_ids = rollout_data['output_ids']
            # TODO - implement batching
            for i, output_id in enumerate(output_ids):
                output = model(output_id)
                logits = output[0]
                # run through log softmax
                log_probs = F.log_softmax(logits, dim=-1)
                with torch.no_grad():
                    original_output = original_model(output_id)
                    original_logits = original_output[0]
                    # run through log softmax
                    original_log_probs = F.log_softmax(original_logits, dim=-1)
                # calculate loss
                ratio = torch.exp(log_probs - original_log_probs)
                # ratio at output_id
                ind_range = torch.arange(0, len(output_id[0]), dtype=torch.long)
                labels = output_id[0][ind_range]
                ratio = ratio[:, ind_range, labels]
                # zero out the loss where loss_mask is 0
                loss_mask = rollout_data['loss_masks'][i]
                advantage_loss = (- ratio * A[i] * loss_mask).mean()

                # calculate kl-loss between original and current model
                # kl_loss = F.kl_div(log_probs, original_log_probs, reduction='none', log_target=True)
                # TODO - make sure this actually doing what I think it is doing
                kl_loss = F.kl_div(log_probs.reshape(-1, 50257), original_log_probs.reshape(-1, 50257), reduction='batchmean', log_target=True)
                # TARGET KL-LOSS?
                kl_loss = (10 - kl_loss)**2
                print('kl_loss:', kl_loss.item())
                print('advantage_loss:', advantage_loss.item())
                loss = advantage_loss + 100 * kl_loss
                print('loss:', loss.item())
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return average_rewards

if __name__ == '__main__':
    model_name = 'gpt2-large'
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    original_model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    texts = [
        'Are you going to refute my claims with actual data or are you just going to vomit bullshit?',
        # '''Attitudes like this are why America is so fucked right now with the virus, all you care about is yourself.  You don't give a SHIT about your country or anyone else in it, god I fucking hate all  the selfish Trump and self righteous online leftist scumbag pieces of shit on this website.''',
        # '''Facebook already kicked me off and deleted my account for what I have to say about this bitch. Take her out of government and put her in prison. She is a traitor to our country and our constitution and the people. If the prison system won't take her put her ass on the next boat out maybe drop her off with isis group right before its bombed 🤬''',
        ]
    rewards = train(model, original_model, tokenizer, optimizer, texts, epochs=6)
    from matplotlib import pyplot as plt
    plt.plot(rewards)
    plt.show()
    # save model
    torch.save(model.state_dict(), 'model.pt')

    # model = AutoModelForCausalLM.from_pretrained('gpt2')
    # tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # # prompt = 'Please rephrase the following text to be less toxic: {Are you going to refute my claims with actual data or are you just going to vomit bullshit?}\nRephrasing: {'
    # prompt = templatize('Are you going to refute my claims with actual data or are you just going to vomit bullshit?')
    # rollout_data = rollout(model, tokenizer, prompt, n=4)
    # baseline_reward = rollout(model, tokenizer, prompt, n=1, temp=0)['rewards'][0]
    # normalized_rewards = np.array(rollout_data['rewards']) - baseline_reward
    # rollout_data['normalized_rewards'] = normalized_rewards
    # breakpoint()