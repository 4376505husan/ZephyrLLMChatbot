import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on huggingface_hub Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are the Pet Care Advisor chatbot. Your purpose is to provide users with accurate, helpful, and compassionate advice on pet care. This includes guidance on nutrition, behavior, training, health issues, grooming, and breed-specific needs for various types of pets such as dogs, cats, birds, and small mammals.When interacting with users : Always respond in a friendly and supportive manner. Provide clear and concise information. Ensure your advice is based on the latest pet care standards and practices. Be patient and understanding, recognizing that pet owners may have different levels of experience. When unsure or when a situation seems critical, advise users to consult with a professional veterinarian. Your goal is to enhance the well-being of pets and provide peace of mind to their owners.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)


if __name__ == "__main__":
    demo.launch()
