import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report
from time import sleep

client = OpenAI(
    api_key= 'your_open_ai_key',
)

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

df = pd.read_csv(f'./data/BoolQ.csv')

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    question = row['question']
    passage = row['passage']

    agent1_Counterfactual_question = f"""Passage: {passage}
Question: {question}
You are a reading comprehension expert. Please fully understand the evidence in passage and question's claim. Why the claim is true? What is the evidence that supports the claim?

Output Format:
The claim is true. Let us reasoning step by step.
First, the claim say that [your understand about claim].
Then, the evidence shows that [content in evidence that support the claim].
So, the claim is true.
"""

    agent1_reply, _ = chat_with_gpt(agent1_Counterfactual_question, [])
    df.loc[idx, 'agent1_reply_1'] = agent1_reply


    agent2_Counterfactual_question = f"""Passage: {passage}
Question: {question}
You are a reading comprehension expert. Please fully understand the evidence in passage and question's claim. Why the claim is false? What is the evidence that refuses the claim?

Output Format:
The claim is false. Let us reasoning step by step.
First, the claim say that [your understand about claim].
Then, the evidence conflicts that [content in evidence that refuses the claim].
So, the claim is false.
"""

    agent2_reply, _ = chat_with_gpt(agent2_Counterfactual_question, [])
    df.loc[idx, 'agent2_reply_1'] = agent2_reply

    critic_1_question = f"""Passage: {passage}
Question: {question}
Assistant: {agent1_reply}

As a critic, your primary role is to meticulously assess the response provided by the assistant. Your task involves identifying any inaccuracies or logical fallacies in the fact checker's analysis and offering detailed feedback that guides the fact checker towards improving their response. 

Instructions:
1. Analyze the assistant's response considering the evidence and claim.
2. Identify errors, discrepancies, or oversights in the analysis.
3. Suggest specific revisions to improve accuracy and address the claim.
4. Provide instructive feedback to aid the assistant's understanding.

Output Format:
- Errors identified: [List errors]
- Suggestions for revision: [Specific actions or corrections]
- Feedback to the assistant: [Instructive prompt]
"""

    critic_1_reply, _ = chat_with_gpt(critic_1_question, [])
    df.loc[idx, 'agent1_reflection'] = critic_1_reply

    critic_2_question = f"""Passage: {passage}
Question: {question}
Assistant: {agent2_reply}

As a critic, your primary role is to meticulously assess the response provided by the assistant. Your task involves identifying any inaccuracies or logical fallacies in the fact checker's analysis and offering detailed feedback that guides the fact checker towards improving their response. 

Instructions:
1. Analyze the assistant's response considering the evidence and claim.
2. Identify errors, discrepancies, or oversights in the analysis.
3. Suggest specific revisions to improve accuracy and address the claim.
4. Provide instructive feedback to aid the assistant's understanding.

Output Format:
- Errors identified: [List errors]
- Suggestions for revision: [Specific actions or corrections]
- Feedback to the assistant: [Instructive prompt]
"""
    critic_2_reply, _ = chat_with_gpt(critic_2_question, [])
    df.loc[idx, 'agent2_reflection'] = critic_2_reply

    revision_1_question = f"""Passage: {passage}
Question: {question}

Assistant: {agent1_reply}

Critic: {critic_1_reply}

Play the role of assistant and revise your response based on the feedback provided by the critic. Your task is to carefully consider the critic's analysis and suggestions, then adjust your response to address any errors or oversights identified by the critic.

Output Format:
Claim: [Reiterate the claim]
Judgement: The claim is [true/false]. 
Explanation: [Provide a brief explanation of your reasoning]
"""

    revision_1_reply, _ = chat_with_gpt(revision_1_question, [])
    df.loc[idx, 'agent1_reply_2'] = revision_1_reply

    revision_2_question = f"""Passage: {passage}
Question: {question}

Assistant: {agent2_reply}

Critic: {critic_2_reply}

Play the role of assistant and revise your response based on the feedback provided by the critic. Your task is to carefully consider the critic's analysis and suggestions, then adjust your response to address any errors or oversights identified by the critic.

Output Format:
Claim: [Reiterate the claim]
Judgement: The claim is [true/false]. 
Explanation: [Provide a brief explanation of your reasoning]
"""

    revision_2_reply, _ = chat_with_gpt(revision_2_question, [])
    df.loc[idx, 'agent2_reply_2'] = revision_2_reply

    judge_question = f"""Passage: {passage}
Question: {question}

Agent 1's Revised Response: {revision_1_reply}
Agent 2's Revised Response: {revision_2_reply}

Your role involves giving the final judgment on the claim based on the revised responses provided by the two agents. Your task is to meticulously assess the consistency, accuracy, and completeness of each response in relation to the evidence provided and the nature of the claim.

Instructions:
1. Review and compare the revised responses from both agents.
2. Evaluate how well each response uses the evidence to support or refute the claim.
3. Consider the logical coherence and factual correctness of each response.
4. Make a final determination on the claim's validity based on the synthesis of the two responses and the evidence.

Output Format:
The claim is [true/false]. 
"""

    judge_reply, _ = chat_with_gpt(judge_question, [])
    df.loc[idx, 'judge'] = judge_reply

df.to_csv(f'./result/CFMAD_result.csv', index=False)