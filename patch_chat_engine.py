import logging
from typing import Any, List, Optional
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import trace_method
from llama_index.core.chat_engine.types import (
    StreamingAgentChatResponse,
)
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    StreamingResponse,
    # AsyncStreamingResponse,
)
from llama_index.core.types import Thread
from llama_index.core.chat_engine.utils import (
    response_gen_from_query_engine,
    # aresponse_gen_from_query_engine,
)

logger = logging.getLogger(__name__)

@trace_method("chat")
def patched_stream_chat(
    self, message: str, chat_history: Optional[List[ChatMessage]] = None
) -> StreamingAgentChatResponse:
    chat_history = chat_history or self._memory.get(input=message)

    # Generate standalone question from conversation context and last message
    condensed_question = self._condense_question(chat_history, message)
    self.condensed_question = condensed_question

    log_str = f"Querying with: {condensed_question}"
    logger.info(log_str)
    if self._verbose:
        print(log_str)

    # TODO: right now, query engine uses class attribute to configure streaming,
    #       we are moving towards separate streaming and non-streaming methods.
    #       In the meanwhile, use this hack to toggle streaming.
    from llama_index.core.query_engine.retriever_query_engine import (
        RetrieverQueryEngine,
    )

    if isinstance(self._query_engine, RetrieverQueryEngine):
        is_streaming = self._query_engine._response_synthesizer._streaming
        self._query_engine._response_synthesizer._streaming = True

    # Query with standalone question
    query_response = self._query_engine.query(condensed_question)

    # NOTE: reset streaming flag
    if isinstance(self._query_engine, RetrieverQueryEngine):
        self._query_engine._response_synthesizer._streaming = is_streaming

    tool_output = self._get_tool_output_from_response(
        condensed_question, query_response
    )

    # Record response
    if (
        isinstance(query_response, StreamingResponse)
        and query_response.response_gen is not None
    ):
        # override the generator to include writing to chat history
        self._memory.put(ChatMessage(role=MessageRole.USER, content=message))
        response = StreamingAgentChatResponse(
            chat_stream=response_gen_from_query_engine(query_response.response_gen),
            sources=[tool_output],
        )
        thread = Thread(
            target=response.write_response_to_history,
            args=(self._memory,),
        )
        thread.start()
    else:
        raise ValueError("Streaming is not enabled. Please use chat() instead.")
    return response