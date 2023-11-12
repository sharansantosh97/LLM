from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
import io
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "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"

load_dotenv()

# # api_key = os.getenv('OPENAI_API_KEY') 
# api_key = st.secrets["OPENAI_API_KEY"]


async def main():
    st.title('Economic and Social Sustainability of Marginalized and Highly Vulnerable communities')
    # 
    async def storeDocEmbeds(file, filename):
    
        reader = PdfReader(file)

        # Extracting the text from the PDF

        corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

        # PDf is split into chunks of 1000 characters with an overlap of 200 characters

        splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)

        # Chunks are generated from the PDF

        chunks = splitter.split_text(corpus)
        
        # used to generate the embeddings for the PDF 
        embeddings = OpenAIEmbeddings(openai_api_key = api_key)

        ## Facebook Faiss is used to generate an efficient similarity search 
        vectors = FAISS.from_texts(chunks, embeddings)
        
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(vectors, f)

        
    async def getDocEmbeds():
        
        # if not os.path.isfile(filename + ".pkl"):
        #     await storeDocEmbeds(file, filename)
        
        with open("allfourmerged.pdf" + ".pkl", "rb") as f:
            global vectores
            vectors = pickle.load(f)
            
        return vectors
    

    async def conversational_chat(query):
        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        # print("Log: ")
        # print(st.session_state['history'])
        return result["answer"]


    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")

    if 'history' not in st.session_state:
        st.session_state['history'] = []


    #Creating the chatbot interface
    st.title("You can ask anything about - Economic and Social Sustainability")

    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    # uploaded_file = st.file_uploader("Choose a file", type="pdf")

    # if uploaded_file is not None:

    #     with st.spinner("Processing..."):
    #     # Add your code here that needs to be executed
    #         uploaded_file.seek(0)
    #         file = uploaded_file.read()
    #         # pdf = PyPDF2.PdfFileReader()
    #         vectors = await getDocEmbeds(io.BytesIO(file), uploaded_file.name)
    #         qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=vectors.as_retriever(), return_source_documents=True)

    vectors = await getDocEmbeds()
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=vectors.as_retriever(), return_source_documents=True)
    st.session_state['ready'] = True

    # st.divider()

    if st.session_state['ready']:

        if 'generated' not in st.session_state:
            # st.session_state['generated'] = ["Welcome! You can now ask any questions regarding " + uploaded_file.name]
            st.session_state['generated'] = ["Welcome! You can now ask any questions regarding - Economic and Social Sustainability"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey!"]

        # container for chat history
        response_container = st.container()

        # container for text box
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="e.g: List Economic Sustainability statements", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = await conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


if __name__ == "__main__":
    asyncio.run(main())