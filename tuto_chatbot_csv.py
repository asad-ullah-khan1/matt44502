import streamlit as st
from streamlit_chat import message
import tempfile
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# Ensure the user provides an API key before proceeding
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your OpenAI API key, sk-",
    type="password")

if user_api_key:  # Only proceed if an API key is entered
    uploaded_file = st.sidebar.file_uploader("upload", type="csv")

    if uploaded_file:  # Process the file only if provided
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Initialize the CSV loader and load the data
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        data = loader.load()

        # Initialize OpenAIEmbeddings with the provided API key
        embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
        
        # Create a vector store from the documents
        vectors = FAISS.from_documents(data, embeddings)

        # Create the conversational retrieval chain with the LLM and retriever
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                temperature=0.0,
                model_name='gpt-3.5-turbo',
                openai_api_key=user_api_key  # Pass the user's API key here as well
            ),
            retriever=vectors.as_retriever()
        )

        # Define the conversational chat function
       
        def conversational_chat(query):
            result = chain.invoke({"question": query, "chat_history": st.session_state['history']})  # Updated method
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]
        
        # Initialize session state if it hasn't been set up already
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]

        # Streamlit UI components for the chat history and text input
        response_container = st.container()
        
        with st.container():
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')

            # If the user sends a message, process it and update the conversation history
            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        # Display chat messages using the Streamlit Chat component
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
else:  # Prompt the user to enter the API key
    st.sidebar.error("Please enter your OpenAI API key.")

# Instruct the user to run the script with Streamlit if not already running
# (e.g., 'streamlit run your_script.py')
