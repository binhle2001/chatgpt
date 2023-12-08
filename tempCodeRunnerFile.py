with gr.Blocks() as chatgpt:
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
# chatgpt.launch(share=True)  # share=True to share the chat publicly
