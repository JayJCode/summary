import json
from typing import Any
import tiktoken
from loguru import logger


class TokenOperator:
    """Represents the extended Tiktoken: TokenOperator.
    Estimates the number of tokens based on tiktoken model.
    Splits the json data into chunks based on logical units and token limit.
    """

    def __init__(self):
        self.tiktoken_tokenizer = tiktoken.get_encoding("o200k_base")
        self.token_limit = 120000

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text"""
        token_length = len(self.tiktoken_tokenizer.encode(text))
        return token_length

    def chunk_data(self, json_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Split the json data into chunks based on logical units"""
        chunks: list[list[dict[str, Any]]] = []
        current_chunk: list[dict[str, Any]] = []
        current_chunk_size: int = 0
        metadata_size: int = 0
        metadata: dict[str, Any] = {}

        def add_to_chunk(item: dict[str, Any], type: str) -> None:
            """
            Function to add sub chunks to chunk if they don't exceed token limit
            If it exceeds on metadata sub chunk, just end current chunk,
            go to next and add sub chunk with metadata.
            If it exceeds on attribute sub chunk, end current chunk,
            and create a new one with lastly used metadata section, to continue on it.
            After there is metadata info in chunk, add sub chunk with attribute.
            """
            nonlocal current_chunk, current_chunk_size, metadata, metadata_size
            item_text = json.dumps(item)
            item_size = self.estimate_tokens(item_text)
            if type == "metadata":
                metadata = item
                metadata_size = item_size

            if current_chunk_size + item_size > self.token_limit:
                chunks.append(current_chunk)
                if type == "attribute":
                    current_chunk = [metadata]
                    current_chunk_size = metadata_size
                else:
                    current_chunk = []
                    current_chunk_size = 0
            current_chunk.append(item)
            current_chunk_size += item_size

        for result in json_data.get("results", []):
            if isinstance(result, dict) and result.items():
                # Add metadata (everything except attributes)
                metadata = {k: v for k, v in result.items() if k != "attributes"}
                add_to_chunk(metadata, "metadata")

                # Add attributes
                for attribute in result.get("attributes", []):
                    add_to_chunk(attribute, "attribute")
            else:
                add_to_chunk(result, "attribute")

        if current_chunk:
            chunks.append(current_chunk)

        chunked_data = [
            {"chunk_id": idx, "total_chunks": len(chunks), "data": chunk}
            for idx, chunk in enumerate(chunks)
        ]

        logger.info(f"Data has been chunked.")
        return chunked_data
