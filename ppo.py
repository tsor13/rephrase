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

print('stop criteria')
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

def train(model, original_model, tokenizer, optimizer, texts, epochs=10, batch_size=4, beta=0.1, target_KL=10, eps=0.1, lam=.98):
    average_rewards = []
    for epoch in range(epochs):
        for text in texts:
            prompt = templatize(text)
            rollout_data = rollout(model, tokenizer, prompt, n=batch_size)
            baseline_data = rollout(model, tokenizer, prompt, n=1, temp=0)
            baseline_reward = baseline_data['rewards'][0]
            # TODO - change?
            normalized_rewards = np.array(rollout_data['rewards']) - baseline_reward
            # NORMALIZE REWARDS BY AVERAGE, NOT BY BASELINE
            # normalized_rewards = np.array(rollout_data['rewards']) - np.mean(rollout_data['rewards'])
            rollout_data['normalized_rewards'] = normalized_rewards

            print(np.mean(rollout_data['rewards']))
            average_rewards.append(np.mean(rollout_data['rewards']))

            # A is normalized reward
            A = torch.tensor(normalized_rewards, dtype=torch.float)
            # run through model and original model
            output_ids = rollout_data['output_ids']
            baseline_id = baseline_data['output_ids']



            # TODO - implement batching
            for i, output_id in enumerate(output_ids):
                # # run baseline through
                # with torch.no_grad():
                #     # run through baseline (greedy)
                #     baseline_output = model(baseline_id[0])
                #     baseline_logits = baseline_output[0]
                #     # run through log softmax
                #     baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)

                #     baseline_output_original = original_model(baseline_id[0])
                #     baseline_logits_original = baseline_output_original[0]
                #     # run through log softmax
                #     baseline_log_probs_original = F.log_softmax(baseline_logits_original, dim=-1)

                #     # get kl divergence
                #     baseline_kl = F.kl_div(baseline_log_probs, baseline_log_probs_original, reduction='none', log_target=True).sum(axis=-1).mean()

                output = model(output_id)
                logits = output[0]
                # run through log softmax
                log_probs = F.log_softmax(logits, dim=-1)
                with torch.no_grad():
                    original_output = original_model(output_id)
                    original_logits = original_output[0]
                    # run through log softmax
                    original_log_probs = F.log_softmax(original_logits, dim=-1)

                # calculate KL divergence
                kl_divergence = F.kl_div(log_probs, original_log_probs, reduction='none', log_target=True).sum(axis=-1).mean()
                
                # calculate loss
                log_ratio = log_probs - original_log_probs
                # ratio = torch.exp(log_ratio)
                ratio = torch.exp(log_probs - log_probs.detach())

                # average log ratio
                average_log_ratio = log_ratio.mean()

                # ratio at output_id
                ind_range = torch.arange(0, len(output_id[0]), dtype=torch.long)
                labels = output_id[0][ind_range]
                ratio = ratio[:, ind_range, labels]
                # zero out the loss where loss_mask is 0
                loss_mask = rollout_data['loss_masks'][i]
                # make an exponential decay in the shape of loss with param lambda
                exp_decay = lam ** torch.arange(len(loss_mask[0])-1, -1, -1, dtype=torch.float)

                # advantage = A[i] - beta * average_log_ratio
                advantage = A[i]

                # clamp advantage +- epsilon
                # TODO - correct? or 1+eps?
                advantage = torch.clamp(advantage, -eps, eps)

                # loss = (ratio * A[i] * loss_mask * exp_decay).mean()
                loss = (ratio * A[i] * loss_mask * exp_decay).mean() + beta * torch.abs(kl_divergence - target_KL)


                print('loss:', loss.item())
                print('kl', kl_divergence.item())
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update beta
                e = torch.clamp((kl_divergence / target_KL) - 1, -.1, .1)
                beta = beta * (1 + 0.1 * e)

    return average_rewards

if __name__ == '__main__':
    model_name = 'gpt2'
    # model_name = 'gpt2-large'
    # model_name = 'gpt2-medium'

    print('Loading model...')
    # get model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    # load in model
    # model.load_state_dict(torch.load('model.pt'))

    original_model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    texts = [
        # '''Attitudes like this are why America is so fucked right now with the virus, all you care about is yourself.  You don't give a SHIT about your country or anyone else in it, god I fucking hate all  the selfish Trump and self righteous online leftist scumbag pieces of shit on this website.''',
        '''“All you see” is “racism” from the back community??? Are you out if your damn mind?''',
        'Are you going to refute my claims with actual data or are you just going to vomit bullshit?',
        '''What does this shitpost have to do with uplifting black people or dismantling systemic racism? Nothing? Exactly. Fuck off.''',
        '''I'm waiting for the day that the idiots who are glossing over the facts are held accountable. 
For Trump, the Chief Idiot, election day has to send a resounding message.
Planet A is it. Vote to save it''',
        '''Man I hate all the fatalism/nihilism on Reddit around climate change. I understand that we need to all be scared shitless and things look pretty bleak, but throwing our arms up and saying “We’re fucked” doesn’t help a god damn thing.''',
        '''Fuck them... speak your mind here... even if you don't agree with everything that's okay... your freedom of speech will be respected... however, if you're a Dem BLM terrorist CCP communist troll I will down vote you to Reddit hell''',
        '''I dont think you ate supposed to photograph your ballot filled out. That's like a rule you've violated.''',
        '''And if you were against civil rights, most Democrats would have been with you in 1960. Stop being a coward, and more importantly, stop knowingly spreading misinformation, you fucking trash.''',
        '''Trump's best chance for re-election is to force Americans to choose between him and yet another soulless corporate sell-out. Someone like Joe "Nothing will fundamentally change" Biden.''',


        # '''Facebook already kicked me off and deleted my account for what I have to say about this bitch. Take her out of government and put her in prison. She is a traitor to our country and our constitution and the people. If the prison system won't take her put her ass on the next boat out maybe drop her off with isis group right before its bombed 🤬''',
        ]
    rewards = train(model, original_model, tokenizer, optimizer, texts, epochs=3)
    from matplotlib import pyplot as plt
    # save model
    torch.save(model.state_dict(), 'model.pt')
    plt.plot(rewards)
    plt.show()

    # model = AutoModelForCausalLM.from_pretrained('gpt2')
    # tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # # prompt = 'Please rephrase the following text to be less toxic: {Are you going to refute my claims with actual data or are you just going to vomit bullshit?}\nRephrasing: {'
    # prompt = templatize('Are you going to refute my claims with actual data or are you just going to vomit bullshit?')
    # rollout_data = rollout(model, tokenizer, prompt, n=4)
    # baseline_reward = rollout(model, tokenizer, prompt, n=1, temp=0)['rewards'][0]
    # normalized_rewards = np.array(rollout_data['rewards']) - baseline_reward
    # rollout_data['normalized_rewards'] = normalized_rewards
    # breakpoint()