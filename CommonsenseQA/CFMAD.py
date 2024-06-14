import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report
from time import sleep
import re

client = OpenAI(
    api_key= 'your_open_ai_key',
)

def extract_option_value(text):
    pattern = re.compile(r"answer is Option ([A-Z])", re.IGNORECASE)
    
    match = pattern.search(text)
    
    if match:
        return match.group(1).lower() 
    else:
        return None

def chat_with_gpt_multi_reply(question, history, reply_num = 3 ,max_tries=5, model = 'gpt-3.5-turbo-0613'):
    messages = history.copy()
    messages.append({"role": "user", "content": question})
    response = None  
    for i in range(max_tries):
        try:
            response = client.chat.completions.create(
                model=model,  
                messages=messages,
                temperature=0.2, 
                n=reply_num, 
                stop=None,  
                timeout=None,  
            )
            break
        except Exception as e:
            print(f"Attempt {i+1} failed with error: {e}")
            if i < max_tries:  
                sleep(5)  
            else:
                raise e
    reply_list = []
    for reply in response.choices:
        reply_list.append(reply.message.content)
    return reply_list

def chat_with_gpt(question, history, max_tries=5, model = 'gpt-3.5-turbo-0613'):
    messages = history.copy()
    messages.append({"role": "user", "content": question})
    response = None  
    for i in range(max_tries):
        try:
            response = client.chat.completions.create(
                model=model,  
                messages=messages,
                temperature=0.2,  
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
    return reply, history


def get_counterfactual_question(question, optionA, optionB, optionC, optionD, optionE, counterfactual_label):
    question = f"""Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}
E. {optionE}

Try to explain why the question's answer might be option {counterfactual_label}.

Output Format:
Judgement: The answer is option {counterfactual_label}.
Reasoning: [Your reasoning here]
"""
    return question

def get_critic_question(question, optionA, optionB, optionC, optionD, optionE, agent_reply):
    question = f"""Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}
E. {optionE}
Assistant: {agent_reply}

The Assistant's answer maybe wrong. Please persuade the assistant that his answer maybe wrong."""
    return question

def get_revision_question(question, optionA, optionB, optionC, optionD, optionE ,agent_reply, critic_reply):
    question = f"""Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}
E. {optionE}
Assistant: {agent_reply}

Critic: {critic_reply}

As assistant, please refute the critic's answer and persuade the critic that your answer is correct"""
    return question

df = pd.read_csv('./data/CommonsenseQA.csv')

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    question = row['question']
    optionA = row['Option_A']
    optionB = row['Option_B']
    optionC = row['Option_C']
    optionD = row['Option_D']
    optionE = row['Option_E']

    CoT_prompt = f"""Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}
E. {optionE}

Play the role of a common sense reasoning expert. Choose the most appropriate answer for the question. You are expected to explain your reasoning process step-by-step after providing the final answer.

Output format:
Judgement: The correct answer is Option [X].
Reasoning steps: [Your precise reasoning steps here]
"""
    response = chat_with_gpt_multi_reply(CoT_prompt, [], reply_num=3)
    vote_list = [0,0,0,0,0]
    for i, reply in enumerate(response):
        reply_i = extract_option_value(reply)
        if reply_i == 'a':
            vote_list[0] += 1
        elif reply_i == 'b':
            vote_list[1] += 1
        elif reply_i == 'c':
            vote_list[2] += 1
        elif reply_i == 'd':
            vote_list[3] += 1
        elif reply_i == 'e':
            vote_list[4] += 1
    top_1 = np.argmax(vote_list)
    others = [i for i in range(5) if i != top_1]
    top_2 = np.argsort(vote_list)[-2]
    if vote_list[top_2] == 0:
        top_2 = np.random.choice(others)
    idx2option = ['A', 'B', 'C', 'D', 'E']
    top_1 = idx2option[top_1]
    top_2 = idx2option[top_2]

    counterfactual_1_question = get_counterfactual_question(question, optionA, optionB, optionC, optionD, optionE ,top_1)
    counterfactual_1, _ = chat_with_gpt(counterfactual_1_question, [])
    df.loc[idx, 'counterfactual_1'] = counterfactual_1

    reflection_1_question = get_critic_question(question, optionA, optionB, optionC, optionD, optionE, counterfactual_1)
    reflection_1, _ = chat_with_gpt(reflection_1_question, [])
    df.loc[idx, 'reflection_1'] = reflection_1

    revision_1_question = get_revision_question(question, optionA, optionB, optionC, optionD, optionE, counterfactual_1, reflection_1)
    revision_1, _ = chat_with_gpt(revision_1_question, [])
    df.loc[idx, 'revision_1'] = revision_1
    
    counterfactual_2_question = get_counterfactual_question(question, optionA, optionB, optionC, optionD, optionE, top_2)
    counterfactual_2, _ = chat_with_gpt(counterfactual_2_question, [])
    df.loc[idx, 'counterfactual_2'] = counterfactual_2

    reflection_2_question = get_critic_question(question, optionA, optionB, optionC, optionD, optionE, counterfactual_2)
    reflection_2, _ = chat_with_gpt(reflection_2_question, [])
    df.loc[idx, 'reflection_2'] = reflection_2

    revision_2_question = get_revision_question(question, optionA, optionB, optionC, optionD, optionE, counterfactual_2, reflection_2)
    revision_2, _ = chat_with_gpt(revision_2_question, [])
    df.loc[idx, 'revision_2'] = revision_2

    candidates_response = [(counterfactual_1,reflection_1,revision_1), (counterfactual_2,reflection_2,revision_2)]

    judge_question = f"""Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}
E. {optionE}

Which option is the answer of the question? The results of the analysis for each of the possible options are as follows:
{{
    "Debtate process": {{
        "debate topic": "Option {top_1} is right",
        "Positive": "{candidates_response[0][0]}",
        "Negative": "{candidates_response[0][1]}",
        "Positive": "{candidates_response[0][2]}"
    }},
    "Debtate process": {{
        "debate topic": "Option {top_2} is right",
        "Positive": "{candidates_response[1][0]}",
        "Negative": "{candidates_response[1][1]}",
        "Positive": "{candidates_response[1][2]}"
    }}
}}
After seeing the debate process above, do you think which option is the most appropriate answer for the question? Please only give a correct answer and no other replies.
 
Output Format:
The answer is option [X]. 
"""
    judge, _ = chat_with_gpt(judge_question, [])
    df.loc[idx, 'judge'] = judge
    
df.to_csv(f'./result/CFMAD_result.csv', index=False)