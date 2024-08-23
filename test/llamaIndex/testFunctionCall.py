from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

def output_structured_data(answer: str):
    
    
output_structured_data_tool = FunctionTool.from_defaults(fn=output_structured_data)

llm = OpenAI(model="gpt-3.5-turbo-1106")
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool, output_structured_data_tool], llm=llm, verbose=True
)

response = agent.stream_chat('''What is llama3 8b's throughput given 
Model,Input,Output,Throughput（requests/s),Throughput（tokens/s),Avg latency（Batch=1）,Avg latency（Batch=8）,Avg latency（Batch=64）
Aquila-7B,60,20,38.19,3055.14,636.76,771.71,1775.86
Aquila-7B,128,20,20.76,3072.22,642.72,882.7,2792.28
Aquila2-7B,60,20,38.2,3056.2,637.89,766.23,1171.65
Aquila2-7B,128,20,20.45,3025.99,645.13,886.26,2805.52
Baichuan-7B,60,20,39.81,3184.63,625.2,753.94,1723.33
Baichuan-7B,128,20,21.34,3158.29,629.15,871.17,2793.17
Baichuan2-7B,60,20,39.83,3186.09,648.05,"771,66",1790.73
Baichuan2-7B,128,20,19.88,2942.16,654.05,881.51,3579.45
Llama2-7B-hf,60,20,42.38,3390.27,611.84,740.53,1687.02
Llama2-7B-hf,128,20,22.13,3274.77,616.09,851.13,2751.64
Llama3-8B,60,20,47.96,3837.01,711.23,811.11,1759.25
Llama3-8B,128,20,24.15,3574.85,717.54,919.62,2687.6
Vicuna-7b,60,20,43.21,3456.9,530.74,736.39,1658.04
Vicuna-7b,128,20,23.07,3414.87,548.45,847.51,2753.68
?''')
print(response)