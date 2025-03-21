from typing import List

from core.schemas.summary import ResponseModel
from core.schemas.metadata_retriever import *
from core.services.summary.base import BaseSummaryAssistant
from core.services.summary.prompts import (
    FINAL_SUMMARY_PROMPT, CHUNK_SUMMARY_PROMPT,
    FINAL_JSON_RESPONSE_PROMPT, CHUNK_JSON_RESPONSE_PROMPT
)
from pydantic import BaseModel


class FinalSummaryAssistant(BaseSummaryAssistant):
    PROMPT = FINAL_SUMMARY_PROMPT

    class ResponseListModel(BaseModel):
        results: List[ResponseModel]

class ChunkSummaryAssistant(BaseSummaryAssistant):
    PROMPT = CHUNK_SUMMARY_PROMPT

    class ResponseListModel(BaseModel):
        results: List[ResponseModel]

class FinalJsonResponseAssistant(BaseSummaryAssistant):
    PROMPT = FINAL_JSON_RESPONSE_PROMPT

    class ResponseListModel(BaseModel):
        results: List[ResponseModel]

class ChunkJsonResponseAssistant(BaseSummaryAssistant):
    PROMPT = CHUNK_JSON_RESPONSE_PROMPT

    class ResponseListModel(BaseModel):
        results: List[ResponseModel]