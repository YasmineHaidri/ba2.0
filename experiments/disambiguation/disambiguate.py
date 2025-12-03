import json

from src.openai_api_call import send_to_llm


user_message_template = """
Your task is to identify and resolve ambiguity in complex questions, ensuring they are clear and unambiguous.
This requires pinpointing elements of the question that could be interpreted in more than one way and refining the
question to ensure a single, clear interpretation.
Approach this task as follows:
Analyze the Question: Read the question thoroughly to identify ambiguous parts. Consider the different ways the 
question could be interpreted based on its current wording.
Clarify the Query: Reformulate the question to eliminate
ambiguity. This may involve specifying details, narrowing down broad terms, or providing additional context to 
guide the interpretation.
Here’s an example of how to complete the task:
For example:
Original Question: Who is the 2024 Summer Olympics table tennis singles champion?
Disambiguated Questions: ["Who is the women's singles champion in table tennis at the 2024 Summer Olympics?",
 "Who is the men's singles champion in table tennis at the 2024 Summer Olympics?"]

Original Question: “What country has the most medals in Olympic history?”
Disambiguated Question: [“What country has the most total medals in Olympic history?”, 
“What country has the most gold medals in Olympic history?”, “What country has the most medals in winter Olympic history?”]

Original Question: When was the baseball team winning the World Series in 2015 created?
Disambiguated Question: ["Which baseball team won the World Series in 2015?"]

**Output Format:**
Return a valid JSON array of strings. 
Do not include any numbering, labels, or additional text — only the array of 1, 2 or maximum 3 (avoid 3 unless really
necessary like in example 2) specific retrieval-ready sub-queries.
If you propose more than one make sure they cover different aspects of specification otherwise if they're the same just 
stick with ONE specific question.

Original Question: {query}
Disambiguated Query:"""

def generate_disambiguated_query(query: str):
    """
    Reformulates a query into its abstracted (step-back) version using predefined prompts.
    """
    user_prompt = user_message_template.format(query=query.strip())

    response = send_to_llm(
        user_prompt=user_prompt,
        model="gpt-4o-mini",
        temperature=0.3
    )
    try:
        sub_queries = json.loads(response)
    except json.JSONDecodeError as e:
        # Show the raw response so you can debug formatting problems
        raise ValueError(
            f"LLM did not return valid JSON for query {query!r}. "
            f"Raw response was:\n{response!r}"
        ) from e

    return sub_queries


if __name__ == "__main__":
    response= generate_disambiguated_query('who sings love will keep us alive by the eagles')
    print (response)

