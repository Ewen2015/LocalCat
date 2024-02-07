import streamlit as st

from transformers import AutoTokenizer, AutoModel


# Load the LLM model and tokenizer
model_name = "ZhipuAI/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)


# Define the chat function
def chat(message):
    input_ids = tokenizer.encode(message, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Streamlit app
def main():
    st.title("LLM Chat App")
    st.write("Welcome to the LLM Chat App! Enter your message below:")

    user_input = st.text_input("User Input", "")
    if st.button("Send"):
        response = chat(user_input)
        st.text_area("LLM Response", value=response, height=200)

if __name__ == "__main__":
    main()