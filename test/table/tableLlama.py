# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="osunlp/TableLlama")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("osunlp/TableLlama")
model = AutoModelForCausalLM.from_pretrained("osunlp/TableLlama")

print(dir(model))