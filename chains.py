from langchain_core.output_parsers import StrOutputParser

def create_extraction_chain(llm, prompt):
    # This uses LCEL: Prompt -> LLM -> String Output
    return prompt | llm | StrOutputParser()

def create_evaluation_chain(llm, prompt):
    return prompt | llm | StrOutputParser()