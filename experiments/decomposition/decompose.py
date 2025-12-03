from experiments.openai_api_call import send_to_llm

user_message_template = ''''
You are an expert at analyzing information-seeking questions for document retrieval.

Your task:
- Decide whether the question is complex or multi-hop and should be decomposed into smaller search queries.
- If decomposition is appropriate, output 2 independent, retrieval-ready sub-queries.
- Each sub-query must be self-contained, explicit, and suitable to issue directly to a retriever.
- If decomposition is not appropriate, return the original question unchanged.
- Do not answer the question — only output the sub-queries separated by a comma.

**Output Format:**
Return a valid JSON array of strings. 
Do not include any numbering, labels, or additional text — only the array.

Examples (for retrieval):

Original Question: Which sport did China win more medals in at the 2024 Summer Olympics: table tennis or badminton?
Sub-queries:
["Number of medals won by China in table tennis at the 2024 Summer Olympics",
 "Number of medals won by China in badminton at the 2024 Summer Olympics"]

Original Question: Were Scott Derrickson and Ed Wood of the same nationality?
Sub-queries:
["Nationality of Scott Derrickson",
 "Nationality of Ed Wood"]

Original Question: Did the SpaceX launch more satellites than Blue Origin in 2023?
Sub-queries:
[ "number of satellites launched by spacex in 2023" ,
" number of satellites launched by Blue Origin in 2023 "]

Original Question: {query}
Sub-queries:
'''

def generate_decomposed_query(query: str) -> str:
    """
    Reformulates a query into its abstracted (step-back) version using predefined prompts.
    """
    user_prompt = user_message_template.format(query=query.strip())

    response = send_to_llm(
        user_prompt=user_prompt,
        model="gpt-4o-mini",
        temperature=0.3
    )
    sub_queries = json.loads(response)

    return sub_queries

if __name__ == "__main__":
   # response = generate_decomposed_query('who sings love will keep us alive by the eagles')
   # print(response)
    response = generate_decomposed_query('difference between single layer perceptron and multilayer perceptron')
    print(response)
