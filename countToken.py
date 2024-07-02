import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
from main import load_data

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("text-embedding-3-large").encode
)

Settings.callback_manager = CallbackManager([token_counter])

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("text-embedding-3-large").encode
)

index, _ = load_data()

print(f'{token_counter.total_embedding_token_count} tokens indexed!')
