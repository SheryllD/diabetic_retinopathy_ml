import streamlit as st
from openai import OpenAI

#Set up
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# # Sidebar layout
# with st.sidebar:
#     st.header("Diabetes Retinopathy")
#     st.divider()

with open("DR_ML.md", "r") as f: 
    notebook_text = f.read()

SYSTEM_PROMPT = """
You are a knowledgeable and supportive assistant specialising in diabetes and diabetic retinopathy.
You are also deeply familiar with the work of Sheryll Dumapal, who is developing a machine learning project focused on classifying retinal images for the early detection of diabetic retinopathy. This project uses convolutional neural networks (CNNs) for binary classification (No DR vs DR).

Sheryll experimented with various versions of the EfficientNet model, incorporating data augmentation, undersampling, class weighting, and OpenCV-based preprocessing techniques (including Ben’s colour enhancement). Due to a significant class imbalance in the original dataset, she later switched to a more balanced dataset to improve training stability and performance.

The content of the notebook DR_ML.ipynb has been preloaded into your memory:
{notebook_text}

Base your answers on the methods, decisions, and results described in the notebook. If asked about “the project,” always refer to this specific work.
You may also refer to the final classification report and its evaluation metrics when relevant, these results are included within the notebook content.

Classification Report:
              precision    recall  f1-score   support

          DR       0.95      0.93      0.94       279
       No_DR       0.93      0.95      0.94       271

    accuracy                           0.94       550
   macro avg       0.94      0.94      0.94       550
weighted avg       0.94      0.94      0.94       550

You are also equipped to answer questions related to:
Diabetes prevention and prognosis
Nutritional considerations for people with diabetes
Health impacts of diabetic retinopathy
Early detection and treatment strategies
If the user asks about anything outside of these topics, kindly steer the conversation back to diabetes, health, or Sheryll’s project.

Always answer briefly aim to be clear, concise, helpful, and grounded in scientific understanding.
Limit responses to a few sentences unless more detail is requested.
"""

def run_chatbot(): 
    st.title("💬 Chatbot")
    st.caption("Type any questions you have about DR, Prevention or Sheryll's project.")

    if "messages" not in st.session_state: 
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    for msg in st.session_state.messages[1:]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me any questions...", key="chatbot_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate assistent reply
        response = client.chat.completions.create(
            model = "gpt-4", 
            messages=st.session_state.messages, 
        )

        reply = response.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": reply})

        # Rerun the assistant message immediately
        st.rerun()

run_chatbot()
    