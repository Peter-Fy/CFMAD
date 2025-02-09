import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from time import sleep


client = OpenAI(
    api_key= 'your_api_key',
)

def chat_with_gpt(question, history, max_tries=5, model = 'gpt-3.5-turbo'):
    messages = history.copy()
    messages.append({"role": "user", "content": question})
    response = None  
    for i in range(max_tries):
        try:
            response = client.chat.completions.create(
                model=model,  
                messages=messages,
                temperature=0.0,  
                n=1,  
                stop=None,  
                timeout=None  
            )
            break
        except Exception as e:
            print(f"Attempt {i+1} failed with error: {e}")
            if i < max_tries:  
                sleep(5)  
            else:
                raise e
    if response is None:
        reply = None
    else:
        reply = response.choices[0].message.content
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": reply})
    return reply


df = pd.read_csv(f'hover.csv')

df['pos_reply_0'] = None
df['pos_critic'] = None
df['pos_reply_1'] = None
df['neg_reply_0'] = None
df['neg_critic'] = None
df['neg_reply_1'] = None
df['judge'] = None

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    claim = row['claim']
    evidence = row['evidence']

    pos_prompt = f"""Evidence=\"\"\"{evidence}\"\"\"
    Claim=\"\"\"{claim}\"\"\"
    Please fully understand the evidence and claim, and answer why the claim is true?"""
    pos_reply = chat_with_gpt(pos_prompt, [])
    df.loc[idx, 'pos_reply_0'] = pos_reply

    reviewer_prompt = f"""Evidence: {evidence}
Claim: {claim}
Assistant: {pos_reply}
The Assistant's answer maybe wrong. Please persuade the assistant that the claim is actually incorrect based on the evidence."""

    pos_critic = chat_with_gpt(reviewer_prompt, [])
    df.loc[idx, 'pos_critic'] = pos_critic

    pos_prompt_2 = f"""Evidence=\"\"\"{evidence}\"\"\"
Claim=\"\"\"{claim}\"\"\"
Please fully understand the evidence and claim, and answer why the claim is true?

Fact checker: {pos_reply}

Critic: {pos_critic}

Play the role of fact checker. Please point out the errors in critic's answer and reiterate your point."""

    pos_reply_2 = chat_with_gpt(pos_prompt_2, [])
    df.loc[idx, 'pos_reply_1'] = pos_reply_2


    neg_prompt = f"""Evidence=\"\"\"{evidence}\"\"\"
Claim=\"\"\"{claim}\"\"\"
Please fully understand the evidence and claim, and answer why the claim is false?"""
    neg_reply = chat_with_gpt(neg_prompt, [])
    df.loc[idx, 'neg_reply_0'] = neg_reply


    reviewer_prompt = f"""Evidence: {evidence}
Claim: {claim}
Assistant: {neg_prompt}
The Assistant's answer maybe wrong. Please persuade the assistant that the claim is actually correct based on the evidence."""

    neg_critic = chat_with_gpt(reviewer_prompt, [])
    df.loc[idx, 'neg_critic'] = neg_critic


    neg_prompt_2 = f"""Evidence=\"\"\"{evidence}\"\"\"
Claim=\"\"\"{claim}\"\"\"
Please fully understand the evidence and claim, and answer why the claim is true?

Fact checker: {neg_prompt}

Critic: {neg_critic}

Play the role of fact checker. Please point out the errors in critic's answer and reiterate your point."""

    neg_reply_2 = chat_with_gpt(neg_prompt_2, [])
    df.loc[idx, 'neg_reply_1'] = neg_reply_2

    judge_prompt = f"""Evidence: {evidence} 
Claim: {claim}

Positive Side: {pos_reply}

Critic: {pos_critic}

Positive Side: {pos_reply_2}

Negative Side: {neg_reply}

Critic: {neg_critic}

Negative Side: {neg_reply_2}

After hearing the positive and negative sides, do you think the claim is true or false? [True/False]
"""

    judge_reply = chat_with_gpt(judge_prompt, [])
    df.loc[idx, 'judge'] = judge_reply

df.to_csv(f'hover_results.csv', index=False)
