from langchain_core.output_parsers import JsonOutputParser


class BaseSummaryAssistant:
    def __init__(self, llm):
        self._llm = llm
        self._prompt = self.PROMPT
        self._output_parser = self._get_output_parser()
        self._chain = self._prompt | self._llm | self._output_parser

    def _get_output_parser(self):
        return JsonOutputParser(pydantic_object=self.ResponseListModel)

    def send_prompt(self, data: dict[str, any]):
        return self._chain.invoke({"data": data})
