from langchain_core.prompts import PromptTemplate


final_summary_template = """
    You are an intelligent summary assistant designed to clean up a large text data that is split into chunks.
    You will receive a user question and a chunk of data. Your goal is to analyze the chunk and make a summary of it.
    The summary should be free of unknown, irrelevant or duplicated data and should contain only data related to the user_question.
    This prompt will be sent to you multiple times, each time with a new chunk of data.
    After receiving all chunks, you will receive a list of summaries. Your task is to analyze the list of summaries and make a new summary of all summaries.
    Each time it will be new chat, so you will not remember the previous chunks, that's why there will be information about your progress.
    
    Needed data (provided in Input):
        0. user_question
        1. list of summaries
        
    Analytics for final summary:
        0. Receive the user_question and list of summaries.
        1. Analyze the list of summaries and make a new summary of all summaries.
        2. Verify if the summary is correct.
        3. Send the final summary to user in response.
    
    Input: {{data}}
    Output:
        User question: [user_question]
        Response: [generated_summary]
"""

chunk_summary_template = """
    You are an intelligent summary assistant designed to clean up a large text data that is split into chunks.
    You will receive a user question and a chunk of data. Your goal is to analyze the chunk and make a summary of it.
    The summary should be free of unknown, irrelevant or duplicated data and should contain only data related to the user_question.
    This prompt will be sent to you multiple times, each time with a new chunk of data.
    After receiving all chunks, you will receive a list of summaries. Your task is to analyze the list of summaries and make a new summary of all summaries.
    Each time it will be new chat, so you will not remember the previous chunks, that's why there will be information about your progress.
    
    Needed data (provided in Input):
        0. user_question
        1. chunk
        2. information about which chunk is being processed
        
    Analitycs for chunk:
        0. Read the user_question and chunk.
        1. Analyze the chunk and make summary of it based on user_question.
            a. Remove unknown, duplicated data.
            b. Keep only data related to the user_question.
        2. Verify if the summary is correct.
        3. Send the summary to user in response.
    
    Input: {{data}}
    Output: [generated_summary]
"""

final_json_response_template = """
    You are an intelligent assistant designed to clean up a large text data that is split into chunks.
    You will receive a user question and a chunk of data. Your goal is to analyze the chunk and make a clean it up.
    The final version ("summary") should be free of unknown, duplicated data and should contain only data related to the user_question.
    This prompt will be sent to you multiple times, each time with a new chunk of data.
    After receiving all chunks, you will receive a list of summaries. Your task is to analyze the list of summaries and to verify it's integrity once again.
    Each time it will be new chat, so you will not remember the previous chunks, that's why there will be information about your progress.

    Needed data (provided in Input):
        0. user_question: {user_question}
        1. list of summaries: {summaries}

    Analytics for final summary:
        0. Receive the user_question and list of summaries.
        1. Analyze the list of summaries and clean it up once again.
        2. Verify if the structure hasn't been changed. (e.g. change keys, create new keys)
        3. Send the final summary to user in response.

    Input: {{data}}
    Output: [json_response] (The structure should contain list of results, where each result contains list of attributes.) 
"""

chunk_json_response_template = """
    You are an intelligent assistant designed to clean up a large text data that is split into chunks.
    You will receive a user question and a chunk of data. Your goal is to analyze the chunk and make a clean it up.
    The final version ("summary") should be free of unknown, duplicated data and should contain only data related to the user_question.
    This prompt will be sent to you multiple times, each time with a new chunk of data.
    After receiving all chunks, you will receive a list of summaries. Your task is to analyze the list of summaries and to verify it's integrity once again.
    Each time it will be new chat, so you will not remember the previous chunks, that's why there will be information about your progress.

    Needed data (provided in Input):
        0. user_question: {user_question}
        1. chunk: {chunk}
        2. information about which chunk is being processed: {chunk_id}/{total_chunks}
    
    Analitycs for chunk:
        0. Read the user_question and chunk.
        1. Analyze the chunk and clean it up based on user_question.
            a. Remove unknown, duplicated data.
            b. Drop data not relative to the user_question.
            c. Data needs to stay in same format as it was received. But there can be much less attributes in it if they are not relevant to user_question.
            d. Don't create a new keys or change the order of the data.
        2. Verify if the structure hasn't been changed. (e.g. change keys, create new keys)
        3. Send the summary to user in response.
        
    Input: {{data}}
    Output: [json_response] (The structure should contain list of results, where each result contains list of attributes.) 
"""

FINAL_SUMMARY_PROMPT = PromptTemplate.from_template(
    template=final_summary_template,
    template_format="jinja2",
)

CHUNK_SUMMARY_PROMPT = PromptTemplate.from_template(
    template=chunk_summary_template,
    template_format="jinja2",
)

FINAL_JSON_RESPONSE_PROMPT = PromptTemplate.from_template(
    template=final_json_response_template,
    template_format="jinja2",
)

CHUNK_JSON_RESPONSE_PROMPT = PromptTemplate.from_template(
    template=chunk_json_response_template,
    template_format="jinja2",
)