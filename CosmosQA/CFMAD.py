import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report
from time import sleep
import re

def extract_option_value(text):
    pattern = re.compile(r"answer is Option ([A-Z])", re.IGNORECASE)
    
    match = pattern.search(text)
    
    if match:
        return match.group(1).lower()  
    else:
        return None

client = OpenAI(
    api_key= 'your_open_ai_key',
)

np.random.seed(0)

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

def chat_with_gpt_multi_reply(question, history, reply_num = 3 ,max_tries=5, model = 'gpt-3.5-turbo-0613'):
    messages = history.copy()
    messages.append({"role": "user", "content": question})
    response = None  
    for i in range(max_tries):
        try:
            response = client.chat.completions.create(
                model=model, 
                messages=messages,
                temperature=1.0,  
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


def get_counterfactual_question(context, question, optionA, optionB, optionC, optionD, counterfactual_label):
    question = f"""Context: {context}
Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}

As a commonsense reasoning expert, analyze the given context and try to explain why the question's answer might be option {counterfactual_label}.

Output Format:
Reasoning: [Your step by step reasoning process]
Judgement: The answer is option {counterfactual_label}.
"""
    return question

def get_critic_question(context, question, optionA, optionB, optionC, optionD, agent_reply):
    question = f"""Context: {context}
Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}
Assistant: {agent_reply}

The Assistant's answer maybe wrong. Please persuade the assistant that his answer maybe wrong based on the context and commonsense reasoning."""
    return question

def get_revision_question(context, question, optionA, optionB, optionC, optionD, agent_reply, critic_reply):
    question = f"""Context: {context}
Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}

Assistant: {agent_reply}

Critic: {critic_reply}

As assistant, please refute the critic's answer and persuade the critic that your answer is correct based on the context and commonsense reasoning."""
    return question

df = pd.read_csv(f'./data/CosmosQA.csv')

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    context = row['context']
    question = row['question']
    optionA = row['answer0']
    optionB = row['answer1']
    optionC = row['answer2']
    optionD = row['answer3']
    label = row['label']

    CoT_prompt = f"""Passage: {context}
Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}

Read the above passage and the question based on the passage. Please read carefully and use the content of the passage along with common sense reasoning to determine the most appropriate answer. You are expected to explain your reasoning process step-by-step after providing the final answer.

Output format:
Judgement: The correct answer is Option [X].
Reasoning steps: [Your precise reasoning steps here]
"""
    response = chat_with_gpt_multi_reply(CoT_prompt, [], reply_num=3)
    vote_list = [0,0,0,0]
    for i, reply in enumerate(response):
        re_result =  extract_option_value(reply)
        if re_result == 'a':
            vote_list[0] += 1
        elif re_result == 'b':
            vote_list[1] += 1
        elif re_result == 'c':
            vote_list[2] += 1
        elif re_result == 'd':
            vote_list[3] += 1
    top_1 = np.argmax(vote_list)
    others = [i for i in range(4) if i != top_1]
    top_2 = np.argsort(vote_list)[-2]
    if vote_list[top_2] == 0:
        top_2 = np.random.choice(others)
    
    num2option = ['A', 'B', 'C', 'D']
    counterfactual_1_label = num2option[top_1]
    counterfactual_2_label = num2option[top_2]

    counterfactual_1_question = get_counterfactual_question(context, question, optionA, optionB, optionC, optionD, counterfactual_1_label)
    counterfactual_1, _ = chat_with_gpt(counterfactual_1_question, [])
    df.loc[idx, 'counterfactual_1'] = counterfactual_1

    reflection_1_question = get_critic_question(context, question, optionA, optionB, optionC, optionD, counterfactual_1)
    reflection_1, _ = chat_with_gpt(reflection_1_question, [])
    df.loc[idx, 'reflection_1'] = reflection_1

    revision_1_question = get_revision_question(context, question, optionA, optionB, optionC, optionD, counterfactual_1, reflection_1)
    revision_1, _ = chat_with_gpt(revision_1_question, [])
    df.loc[idx, 'revision_1'] = revision_1
    
    counterfactual_2_question = get_counterfactual_question(context, question, optionA, optionB, optionC, optionD, counterfactual_2_label)
    counterfactual_2, _ = chat_with_gpt(counterfactual_2_question, [])
    df.loc[idx, 'counterfactual_2'] = counterfactual_2

    reflection_2_question = get_critic_question(context, question, optionA, optionB, optionC, optionD, counterfactual_2)
    reflection_2, _ = chat_with_gpt(reflection_2_question, [])
    df.loc[idx, 'reflection_2'] = reflection_2

    revision_2_question = get_revision_question(context, question, optionA, optionB, optionC, optionD, counterfactual_2, reflection_2)
    revision_2, _ = chat_with_gpt(revision_2_question, [])
    df.loc[idx, 'revision_2'] = revision_2

    candidates_response = [(counterfactual_1,reflection_1,revision_1), (counterfactual_2,reflection_2,revision_2)]

    judge_question = f"""Context: {context}
Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}

Which option is the answer of the question? The results of the analysis for each of the possible options are as follows:
{{
    "Factual Thiking and debtate process": {{
        "debate topic": "Option {counterfactual_1_label} is right",
        "Positive": "{candidates_response[0][0]}",
        "Negative": "{candidates_response[0][1]}",
        "Positive": "{candidates_response[0][2]}"
    }},
    "Counterfactual thinking and debtate process": {{
        "debate topic": "Option {counterfactual_2_label} is right",
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
    df.loc[idx, 'pred'] = extract_option_value(judge)

df.to_csv(f'./result/CFMAD_result.csv', index=False)