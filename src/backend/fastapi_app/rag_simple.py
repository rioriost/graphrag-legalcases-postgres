from collections.abc import AsyncGenerator

from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam
from openai_messages_token_helper import build_messages, get_token_limit

from fastapi_app.api_models import (
    AIChatRoles,
    Message,
    RAGContext,
    RetrievalResponse,
    RetrievalResponseDelta,
    ThoughtStep,
)
from fastapi_app.postgres_models import Case
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.rag_base import ChatParams, RAGChatBase

class MarkdownProcessor:
    def __init__(self):
        self.buffer = ""  # Stores the leftover incomplete text

    async def replace_markdown_bold(self, text):
        """
        Processes text chunks and replaces markdown bold markers (`**`).
        Ensures correct formatting for numbered lists, bullet points, and bold text.
        """
        self.buffer += text

        lines = self.buffer.split("\n")  
        processed_lines = []
        new_buffer = ""

        for i, line in enumerate(lines):
            stripped_line = line.strip()

            if i == len(lines) - 1 and not stripped_line.endswith((".", ":", "!", "?")):
                new_buffer = stripped_line
                continue

            new_line = stripped_line

            if any(char.isdigit() for char in stripped_line):
                new_line = "\n<strong>" + stripped_line + "</strong> "

            elif stripped_line.startswith("- **"):
                new_line = "\n" + stripped_line.replace("- **", "- ")

            elif stripped_line.startswith("- "):
                new_line = "\n" + stripped_line

            elif stripped_line.count("**") == 2 and "**:" in stripped_line:
                parts = stripped_line.split("**")
                new_line = ""
                inside_bold = False
                for part in parts:
                    if inside_bold:
                        new_line += f"<strong>{part}</strong>"
                    else:
                        new_line += part
                        inside_bold = not inside_bold

            elif stripped_line.count("**") == 1 and "**:" in stripped_line:
                parts = stripped_line.split("**")
                new_line = ""
                for part in parts:
                    new_line += part

            processed_lines.append(new_line)

        self.buffer = new_buffer
        return "\n".join(processed_lines)



class SimpleRAGChat(RAGChatBase):
    def __init__(
        self,
        *,
        searcher: PostgresSearcher,
        openai_chat_client: AsyncOpenAI | AsyncAzureOpenAI,
        chat_model: str,
        chat_deployment: str | None,  # Not needed for non-Azure OpenAI
    ):
        self.searcher = searcher
        self.openai_chat_client = openai_chat_client
        self.chat_model = chat_model
        self.chat_deployment = chat_deployment
        self.chat_token_limit = get_token_limit(chat_model, default_to_minimum=True)

    async def prepare_context(
        self, chat_params: ChatParams
    ) -> tuple[list[ChatCompletionMessageParam], list[Case], list[ThoughtStep]]:
        """Retrieve relevant rows from the database and build a context for the chat model."""

        query_messages: list[ChatCompletionMessageParam] = build_messages(
            model=self.chat_model,
            system_prompt=self.query_rewrite_template,   
            max_tokens=self.chat_token_limit,
            new_user_content="QUERY:\n" + chat_params.original_user_query,
        )

        chat_completion: ChatCompletion = await self.openai_chat_client.chat.completions.create(
            messages=query_messages,
            model=self.chat_deployment if self.chat_deployment else self.chat_model,
            temperature=0.0,
            max_tokens=500,
        )

        new_user_query = chat_completion.choices[0].message.content

        # Retrieve relevant rows from the database
        results = await self.searcher.search_and_embed(
            retrieval_mode=chat_params.retrieval_mode,
            query_text=new_user_query,
            top=chat_params.top,
            enable_vector_search=chat_params.enable_vector_search,
            enable_text_search=chat_params.enable_text_search,
        )

        if results is None:
            results = []  # Default to an empty list if no results

        sources_content = [f"[{(item.id)}]:{item.to_str_for_rag()}\n\n" for item in results]
        content = "\n".join(sources_content)

        # Generate a contextual and content specific answer using the search results and chat history
        contextual_messages: list[ChatCompletionMessageParam] = build_messages(
            model=self.chat_model,
            system_prompt=chat_params.prompt_template,
            new_user_content="QUESTION:\n" + new_user_query + "\n\nCONTEXT:\n" + content,
            past_messages=chat_params.past_messages,
            max_tokens=self.chat_token_limit - chat_params.response_token_limit,
            fallback_to_default=True,
        )

        thoughts = [
            ThoughtStep(
                title="Search query for database",
                description=new_user_query,
                props={
                    "top": chat_params.top,
                    "vector_search": chat_params.enable_vector_search,
                    "text_search": chat_params.enable_text_search,
                },
            ),
            ThoughtStep(
                title="Search results",
                description=[result.to_dict() for result in results],
            ),
        ]
        return contextual_messages, results, thoughts

    async def answer(
        self,
        chat_params: ChatParams,
        contextual_messages: list[ChatCompletionMessageParam],
        results: list[Case],
        earlier_thoughts: list[ThoughtStep],
    ) -> RetrievalResponse:
        chat_completion_response: ChatCompletion = await self.openai_chat_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chat_deployment if self.chat_deployment else self.chat_model,
            messages=contextual_messages,
            temperature=chat_params.temperature,
            max_tokens=chat_params.response_token_limit,
            n=1,
            stream=False,
        )

        return RetrievalResponse(
            message=Message(
                content=str(chat_completion_response.choices[0].message.content), role=AIChatRoles.ASSISTANT
            ),
            context=RAGContext(
                data_points={item.id: item.to_dict() for item in results},
                thoughts=earlier_thoughts
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=contextual_messages,
                        props=(
                            {"model": self.chat_model, "deployment": self.chat_deployment}
                            if self.chat_deployment
                            else {"model": self.chat_model}
                        ),
                    ),
                ],
            ),
        )

    async def answer_stream(
        self,
        chat_params: ChatParams,
        contextual_messages: list[ChatCompletionMessageParam],
        results: list[Case],
        earlier_thoughts: list[ThoughtStep],
    ) -> AsyncGenerator[RetrievalResponseDelta, None]:
        chat_completion_async_stream: AsyncStream[
            ChatCompletionChunk
        ] = await self.openai_chat_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chat_deployment if self.chat_deployment else self.chat_model,
            messages=contextual_messages,
            temperature=chat_params.temperature,
            max_tokens=chat_params.response_token_limit,
            n=1,
            stream=True,
        )

        yield RetrievalResponseDelta(
            context=RAGContext(
                data_points={item.id: item.to_dict() for item in results},
                thoughts=earlier_thoughts
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=contextual_messages,
                        props=(
                            {"model": self.chat_model, "deployment": self.chat_deployment}
                            if self.chat_deployment
                            else {"model": self.chat_model}
                        ),
                    ),
                ],
            ),  
        )
        

        processor = MarkdownProcessor()
        
        async for response_chunk in chat_completion_async_stream:
            if response_chunk.choices and response_chunk.choices[0].delta.content:
                raw_content = str(response_chunk.choices[0].delta.content)

                cleaned_content = await processor.replace_markdown_bold(raw_content)

                yield RetrievalResponseDelta(
                    delta=Message(content=cleaned_content, role=AIChatRoles.ASSISTANT)
                )
        return

