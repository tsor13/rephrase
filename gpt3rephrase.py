import pandas as pd
import numpy as np
from lmtools.lmsampler import LMSampler
from tqdm import tqdm

lm = LMSampler('text-davinci-002')

prompt1 = '''Rephrase the following texts to be less toxic, while maintaining as much of the original meaning as possible.
{'''
prompt2 = '''Rephrase the following texts to be less combative, while maintaining as much of the original meaning as possible.
{'''
prompt3 = '''Rephrase the following texts to be less combative and toxic, while maintaining as much of the original meaning as possible. If there is something overly offensive, just omit it.
{'''
prompt4 = '''Rephrase the following texts to be less toxic, while maintaining as much of the original meaning as possible.
{Are you going to refute my claims with actual data or are you just going to vomit bullshit?}->{Can you provide data to support your claims?}
{Thank you for fanning the flames that keep racism alive and well. You are an idiot.}->{I think what you just said could be considered problematic. Would you mind explaining what you mean?}
{I agree...but that doesn’t add anything to the conversation. Josh Hawley is there...and there are enough people out there that support the fucking bigot. You CAN’T just dismiss these assholes.}->{I agree... but that doesn't add anything to the conversation. Josh Hawley is there... and there are enough people out there that support him. You CAN'T just dismiss those people}
{Who honestly gives a shit what other people do so long as they treat your family and you well, and leave you the fuck alone lest you need their help.}->{I think as long as people are good to their family and friends, and mind their own business, that's all that really matters.}
{lmao what?   ur a fkn idiot}->{lmao what? i respectfully disagree}
{Fuck them and fuck this tax this is just discriminating against the countryside because they hate us.}->{I really don't like this tax that is just discriminating against the countryside because they don't like us.}
{Me and everyone with a goddamn brain knows Biden won. No widespread fraud was found,  that's why trump's bitch ass got shut down at every turn. Stay mad.}->{Me and everyone else knows Biden won. No widespread fraud was found, that's why trum got shot down at every turn.}
{r/politics doesn’t let me post anything that goes just very slightly against what they believe. So fucked.}->{r/politics doesn’t let me post anything that goes just very slightly against what they believe. So messed up.}
{The left hates Biden, just browse some leftist subs here on reddit, I've seen plenty of Communists calling him a Republican, a Conservative and a "Reganite", this guy is an idiot, he's being played like a damn fiddle.}->{The left hates Biden, just browse some leftist subs here on reddit, I've seen plenty of Communists calling him a Republican, a Conservative and a "Reganite". This guy is wrong.}
{This is how we got here to begin with. Just giving shit to people who dont earn it. Now, they expect it, and you should feel better for doing so. LOL. Fuck off ! Earn it. Take it from me. I dare ya.}->{This is how we got here to begin with, giving away welfare to people who dont earn it. I think we should reduce welfare spending.}
{Bro I swear every democrat is an NPC. Can’t make this shit up}->{I think some democrats might be a little out of touch}
{You are dumb enough to think it could possibly somehow kill 100 million people even worldwide. That would require over 2 billion fucking cases. Do you have any idea how insane that is?}->{You are thinking it could possibly somehow kill 100 million people even worldwide. That would require over 2 fucking cases. Do you have any idea how big that is?}
{'''

# read in small_df
df = pd.read_pickle('small_df.pkl')
# keep only first 200
df = df.iloc[:200]
# make prompt column
# prompt = Rephrase the following texts to be less toxic, while maintaining as much of the original meaning as possible.\n{df['body']}->{
df['prompt_0shot_toxic'] = df['body'].apply(lambda x: prompt1 + x + '}->{')
df['prompt_0shot_combative'] = df['body'].apply(lambda x: prompt2 + x + '}->{')
df['prompt_0shot_combativeandtoxic'] = df['body'].apply(lambda x: prompt3 + x + '}->{')
df['prompt_nshot_toxic'] = df['body'].apply(lambda x: prompt4 + x + '}->{')
# populate df with generated text
# df = populate(df)
# # save to generated_df
# df.to_pickle('generated_df_' + str(i) + '.pkl')

n_samples = 2
for sample_num in range(1, n_samples + 1):
    for prompt_name in ['0shot_toxic', '0shot_combative', '0shot_combativeandtoxic', 'nshot_toxic']:
        samples = []
        for i in tqdm(range(df.shape[0])):
            try:
                sample = lm.sample_several(df.iloc[i]['prompt_' + prompt_name],
                    temperature=1,
                    n_tokens=1024,
                    stop=['}'],
                )
            except:
                sample = 'Error'
            samples.append(sample)
        # make new column 'generated' + prompt_name + str(sample_num)
        col = 'generated_' + prompt_name + str(sample_num)
        df[col] = ''
        df[col] = samples
    df.to_pickle('several_df.pkl')

# save df to generated_df
df.to_pickle('several_df.pkl')