from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

def send_to_llm( user_prompt: str,
                model: str = "gpt-4o-mini",
                temperature: float = 0.3,
                max_tokens: int = 150) -> str:
    """
    Generic helper function to send a prompt to the OpenAI API.

    Args:
        system_prompt (str): Instruction that sets the assistant's behavior.
        user_prompt (str): The actual user message or query content.
        model (str): Model name to use (default: 'gpt-4o-mini').
        temperature (float): Sampling temperature for creativity control.
        max_tokens (int): Maximum tokens to generate.

    Returns:
        str: The LLM-generated response text.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()
