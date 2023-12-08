import ast  # for converting embeddings saved as strings back to arrays
import configparser
import os
import time

import gradio as gr
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from ai_configs import (
    EMBEDDING_MODEL,
    FILEPATH_EMBEDDINGS,
    INTRODUCTION_MESSAGE,
    MODEL_NAME,
    SYSTEM_CONTENT,
    TOKEN_BUDGET,
)
from scipy import spatial  # for calculating vector similarities for search

env = configparser.ConfigParser()
env.read(".env")
os.environ["OPENAI_API_KEY"] = env["OpenAI"]["OPENAI_KEY_TTL"]
openai.api_key = os.environ["OPENAI_API_KEY"]

model_name = MODEL_NAME
# Read embbeding file
embedding_data = pd.read_csv(FILEPATH_EMBEDDINGS)
# Convert embeddings from CSV str type back to list type
embedding_data["embedding"] = embedding_data["embedding"].apply(
    ast.literal_eval,
)
print("Finished loading embedding data!")


# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 3,
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses,
    sorted from most related to least.
    """
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)

    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = MODEL_NAME) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int,
) -> str:
    """Return a message for GPT,
    with relevant source texts pulled from a dataframe.
    """
    strings, _ = strings_ranked_by_relatedness(query, df)

    """ example:
    #strings, relatednesses = strings_ranked_by_relatedness(
    #     "what solutions that TT provides?",
    #     df,
    #     top_n=5,
    #     )
    #for string, relatedness in zip(strings, relatednesses):
    #    print(f"{relatedness=:.3f}\n{string}\n")
    """

    question = f"\n\nQuestion: {query}"
    message = INTRODUCTION_MESSAGE
    for string in strings:
        next_article = f"\nTT article section:\n--\n{string}\n--"
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def get_response(
    query: str,
    df: pd.DataFrame,
    model: str = MODEL_NAME,
    token_budget: int = TOKEN_BUDGET,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of
    relevant texts and embeddings.
    """
    message = query_message(query, df, model=model, token_budget=token_budget)

    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": message},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    response_message = response["choices"][0]["message"]["content"]
    print(f'Total used tokens: {response["usage"]["total_tokens"]}')
    response_message = response_message.replace("help/document/", "wiki/1-")
    response_message = response_message.replace(">>", ">")
    print("1 ====", response_message, message)
    return response_message, message

message = "askjdfaqskfaqsfn fiqfoiqw fiqwfhoqiw rjfqwjrqw"
response, _ = get_response(
            query=message,
            df=embedding_data,
            model="gpt-3.5-turbo",
        )

# Code for getting chatbot's response ends here. Below code is for UI only.
def format_response(responses: dict):
    """
    (Optional) Format one or multiple responses from version(s) of chatbot

        Parameters:
            responses (dict): chatbot response with the name of model

        Returns:
            output (str): formatted reponse
    """
    output = ""
    for response in responses:
        output += response + (responses[response]) + "\n\n"
    return output




# with gr.Blocks() as chatgpt:
#     chatbot = gr.Chatbot(label="Teamhub", height=500)
#     message = gr.Textbox(
#         label="Enter your chat here",
#         placeholder="Press enter to send a message",
#         show_copy_button=True,
#     )
#     radio = gr.Radio(
#         [
#             "Full model (most capable but slow & expensive)",
#             "Lite model (Capable but fast & cheap)",
#         ],
#         label="Choose a chatbot model",
#         value="Lite model (Capable but fast & cheap)",
#     )
#     clear = gr.Button("Clear all chat")

#     def choice_model(choice):
#         if choice == "Full model (most capable but slow & expensive)":
#             return "gpt-4"
#         else:
#             return "gpt-3.5-turbo"

#     def get_user_message(user_message, history):
#         return "", history + [[user_message, None]]

#     def show_response(history, model):
#         message = history[-1][0]
#         model = choice_model(model)
#         print(f"model: {model}")
#         # Get the response from OpenAI
#         response, _ = get_response(
#             query=message,
#             df=embedding_data,
#             model=model,
#         )

#         # Correct URL
#         # I will remove this function after BE/FE fixing this bug
#         response = response.replace("help/document/", "wiki/1-")
#         response = response.replace(">>", ">")
#         print("Q: ", message, "\nA: ", response, "\n")

#         # Format the response
#         # responses = {
#         #    f"[{MODEL_NAME}] → ": response,
#         # }
#         # response = format_response(responses)
        
#         history[-1][1] = ""
#         for character in response:
#             history[-1][1] += character
#             time.sleep(0.01)
#             yield history

#     message.submit(
#         get_user_message,
#         [message, chatbot],
#         [message, chatbot],
#         queue=False,
#     ).then(
#         show_response,
#         [chatbot, radio],
#         chatbot,
#     )
#     clear.click(lambda: None, None, chatbot, queue=False)
# chatgpt.queue()
# chatgpt.launch(share=True)  
