from core.services.chat.llm import LLMService
from core.services.token_operator import TokenOperator
from loguru import logger
import json
from core.services.summary.summary_assistant import ChunkSummaryAssistant, FinalSummaryAssistant, ChunkJsonResponseAssistant, FinalJsonResponseAssistant


class Summarizer:
    """
    Summarizer class is created to shorten the text data, from metadata service response, to avoid the token limit.
    
    First approach using summary_assistant.
    Take input of chunks, analiz each one, making summary of it.
    Gives summarized commit of what's inside chunks. (string format)

    Second approach using json_response_assistant.
    Takes input of chunks at it is, removes irrelevant attribute, doesn't modify the structure.
    Gives only relevant attributes at the end. (json format)
    """
    def __init__(self):
        self.llm_service = LLMService()
        self.tokenizer = TokenOperator()
        self.chunk_summary_assistant = ChunkSummaryAssistant(self.llm_service)
        self.final_summary_assistant = FinalSummaryAssistant(self.llm_service)
        self.chunk_json_response_assistant = ChunkJsonResponseAssistant(self.llm_service)
        self.final_json_response_assistant = FinalJsonResponseAssistant(self.llm_service)

    def _prepare_data(self, chunks: list[dict[str, any]], user_question: str) -> list[dict[str, any]]:
        """
        Prepare the data_list for sending to catalyst.
        """
        data_list = []
        for chunk in chunks:
            data_list.append({
                "user_question": user_question,
                "chunk": chunk['data'],
                "chunk_id": json.dumps(['chunk_id']),
                "total_chunks": chunk['total_chunks']
            })
        return data_list

    def _send_chunks_summary(self, data_list: list[dict[str, any]]) -> list[str]:
        """
        Send the chunks to make a list of summaries.
        """
        summaries = []
        for data_id, data in data_list:
            summary = self.chunk_summary_assistant.send_prompt(data)
            logger.info("Finished summarizing chunk: {data_id}".format(data_id=data_id))
            summaries.append(summary)
        return summaries

    def _send_chunks_json_response(self, data_list: list[dict[str, any]]) -> list[str]:
        """
        Send the chunks to make a list cleaned up data in json format.
        """
        summaries = []
        for data_id, data in data_list:
            summary = self.chunk_json_response_assistant.send_prompt(data)
            logger.info("Finished summarizing chunk: {data_id}".format(data_id=data_id))
            summaries.append(summary)
        return summaries

    def summarize(self, user_question: str, metadata_response: dict[str, any]) -> str:
        """
        Summarize the metadata response based on user_question.
        That is the main function of the Summarizer class.
        """
        # Chunk data using TokenOperator
        chunked_data = self.tokenizer.chunk_data(metadata_response)
        # Prepare data for AI model
        prepared_data_list = self._prepare_data(chunked_data, user_question)

        # First approach - summary_assistant

        # Send the chunks to catalyst
        summaries = self._send_chunks_summary(prepared_data_list)
        # Final summary
        final_summary = self.final_summary_assistant.send_prompt({
            "user_question": user_question,
            "summaries": summaries
        })

        # # Second approach - json_response_assistant
        #
        # # Send the chunks to catalyst
        # summaries = self._send_chunks_json_response(prepared_data_list)
        # # Final summary
        # final_summary = self.final_json_response_assistant.send_prompt({
        #     "user_question": user_question,
        #     "summaries": summaries
        # })

        logger.info(f"Finished summarizing data")

        return final_summary


# # Find the index of the first occurrence of # and slice the string from there
# for char in final_summary:
#     if char == "=":
#         final_summary = final_summary[final_summary.index(char)+1:]
#         break
#
# # Find the index of the first occurrence of { and slice the string from there
# for char in (final_summary):
#     if char == "{":
#         final_summary = final_summary[:final_summary.index(char) - 18]
#         break


# # Find the index of the first occurrence of [ and slice the string from there
# for char in final_summary:
#     if char == "[":
#         final_summary = final_summary[final_summary.index("["):]
#         break
#
# # Find the index of the last occurrence of ] and slice the string up to there
# for char in reversed(final_summary):
#     if char == "]":
#         final_summary = final_summary[:final_summary.rindex("]") + 1]
#         break
#
# # Remove new lines and backslashes
# final_summary = final_summary.replace("\\n", "").strip()