from experiments.openai_api_call import send_to_llm

# define messages once (outside the function)

user_message_template = """
Your task is to step back and paraphrase a question to a more generic step-back question, 
which is easier to answer.
Output only a single, natural-language question.
Here are a few examples:

Original Question: Jan Šindel’s was born in what country?
Step-Back (Abstracted) Question: What is Jan Šindel’s personal history?

Original Question: When was the abolishment of the studio that distributed *The Game*?
Step-Back (Abstracted) Question: Which studio distributed *The Game*?

Original Question: When was the baseball team winning the World Series in 2015 created?
Step-Back (Abstracted) Question: Which baseball team won the World Series in 2015?

Original Question: What city is the person who broadened the doctrine of philosophy of language from?
Step-Back (Abstracted) Question: Who broadened the doctrine of philosophy of language?

Original Question: {query}
Step-Back (Abstracted) Question:"""

def generate_abstracted_query(query: str) -> str:
    """
    Reformulates a query into its abstracted (step-back) version using predefined prompts.
    """
    user_prompt = user_message_template.format(query=query.strip())

    response = send_to_llm(
        user_prompt=user_prompt,
        model="gpt-4o-mini",
        temperature=0.3
    )

    return response


if __name__ == "__main__":
    response= generate_abstracted_query('who sings love will keep us alive by the eagles')
    print (response)

