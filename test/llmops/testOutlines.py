import re
import numpy as np
import outlines
import streamlit as st
from outlines import generate
import outlines.models as models
from openai import AzureOpenAI

api_version = "2024-05-01-preview"
azure_api_key = st.secrets.azure_api_key
azure_endpoint = st.secrets.azure_endpoint
azure_chat_deployment = st.secrets.azure_chat_deployment
azure_embedding_deployment = st.secrets.azure_embedding_deployment
azure_openai = AzureOpenAI( 
    api_key=azure_api_key,  
    api_version=api_version,
    azure_endpoint = azure_endpoint,
)
assistant = azure_openai.beta.assistants.create(
    name="Data Visualization",
    instructions=f"You are a helpful AI assistant who makes accurate analysis based on data." 
    f"You have access to a sandboxed environment for writing and testing code."
    f"When you are asked to create a visualization you should follow these steps:"
    f"1. Write the code."
    f"2. Anytime you write new code display a preview of the code to show your work."
    f"3. Run the code to confirm that it runs."
    f"4. If the code is successful display the visualization."
    f"5. If the code is unsuccessful display the error message and try to revise the code and rerun going through the steps from above again.",
    tools=[{"type": "code_interpreter"}, {"type": "file_search"}],
    model=azure_chat_deployment #You must replace this value with the deployment name for your model.
)

model = models.openai("gpt-3.5-turbo")

# prompt = """You are a cuisine-identification assistant.
# What type of cuisine does the following recipe belong to?

# Recipe: This dish is made by stir-frying marinated pieces of chicken, vegetables, and chow mein noodles. The ingredients are stir-fried in a wok with soy sauce, ginger, and garlic.

# """
# answer = generate.choice(model, ["Chinese", "Italian", "Mexican", "Indian"])(prompt)
# print(answer)

question = "compare llama with vicuna" #"When I was 6 my sister was half my age. Now I’m 70 how old is my sister?"

csv = '''
,0,1,2,3,4,5,6,7
0,Model,Input,Output,Throughput（requests/s),Throughput（tokens/s),Avg latency（Batch=1）,Avg latency（Batch=8）,Avg latency（Batch=64）
1,Aquila-7B,60,20,38.19,3055.14,636.76,771.71,1775.86
2,Aquila-7B,128,20,20.76,3072.22,642.72,882.7,2792.28
3,Aquila2-7B,60,20,38.2,3056.2,637.89,766.23,1171.65
4,Aquila2-7B,128,20,20.45,3025.99,645.13,886.26,2805.52
5,Baichuan-7B,60,20,39.81,3184.63,625.2,753.94,1723.33
6,Baichuan-7B,128,20,21.34,3158.29,629.15,871.17,2793.17
7,Baichuan2-7B,60,20,39.83,3186.09,648.05,"771,66",1790.73
8,Baichuan2-7B,128,20,19.88,2942.16,654.05,881.51,3579.45
9,Llama2-7B-hf,60,20,42.38,3390.27,611.84,740.53,1687.02
10,Llama2-7B-hf,128,20,22.13,3274.77,616.09,851.13,2751.64
11,Llama3-8B,60,20,47.96,3837.01,711.23,811.11,1759.25
12,Llama3-8B,128,20,24.15,3574.85,717.54,919.62,2687.6
13,Vicuna-7b,60,20,43.21,3456.9,530.74,736.39,1658.04
14,Vicuna-7b,128,20,23.07,3414.87,548.45,847.51,2753.68
'''

@outlines.prompt
def few_shots(question, examples):
    """
    {% for example in examples %}
    Q: {{ example.question }}
    A: {{ example.answer }}
    {% endfor %}
    Q: {{ question }}
    A:
    """

examples = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.",
    },
]
generator = outlines.generate.text(model)

'''
excel_file_path = '/Users/keltonzhang/Code/qc/a/模型性能对比.xlsx'
vector_store = azure_openai.beta.vector_stores.create(name="Model Performances")
# Upload the user provided file to OpenAI
message_file = azure_openai.files.create(
    file=open(excel_file_path, "rb"), purpose="assistants"
)

print(f'provided {message_file.id} to openai')
openai_assistant = azure_openai.beta.assistants.update(
assistant_id=azure_openai.id,
tool_resources={
    "file_search": {
        "vector_store_ids": [vector_store.id]
    },
    "code_interpreter": {
        "file_ids": [message_file.id]
    }
},
)
'''
prompt = question+"\n"+csv #few_shots(question, examples)
answers = generator(prompt, samples=3)

digits = []
for answer in answers:
    try:
        match = re.findall(r"\d+", answer)[-1]
        if match is not None:
            digit = int(match)
            digits.append(digit)
    except AttributeError:
        print(f"Could not parse the completion: '{answer}'")

unique_digits, counts = np.unique(digits, return_counts=True)
results = {d: c for d, c in zip(unique_digits, counts)}
print(results)

max_count = max(results.values())
answer_value = [key for key, value in results.items() if value == max_count][0]
total_count = sum(results.values())
print(
    f"The most likely answer is {answer_value} ({max_count/total_count*100}% consensus)"
)