import streamlit as st
from docx import Document
from io import BytesIO
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import time

genai.configure(api_key="AIzaSyB_OHmnzesSf4O1FeM_9PDFim9N9wPmZqg")

if 'embedding_model' not in st.session_state:
    try:
        model_name = "all-MiniLM-L6-v2"
        st.session_state.embedding_model = SentenceTransformer(model_name)
        st.info(f"Embedding model '{model_name}' loaded successfully.")
    except Exception as e:
        st.error(f"Error loading SentenceTransformer embedding model: {e}")
        st.stop() # Stop if the embedding model can't load

if 'llm_model' not in st.session_state:
    try:
        st.session_state.llm_model=genai.GenerativeModel("gemini-2.0-flash")
        st.info("Gemini LLM model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading Gemini LLM model: {e}")
        st.stop() # Stop if the LLM model can't load


if 'messages' not in st.session_state:
    st.session_state.messages=[]

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index=None

if "text_chunks" not in st.session_state:
    st.session_state.text_chunks=None

if "chunks_embeddings" not in st.session_state:
    st.session_state.chunks_embeddings=None

st.title("Document Question Answering BOT")

uploaded_file= st.file_uploader("Upload your PDF and MS Word File here", type=['docx','pdf'])

if uploaded_file is not st.session_state.get("last_uploaded_file"):
    st.session_state.messages=[]
    st.session_state.faiss_index=None
    st.session_state.text_chunks=None
    st.session_state.chunks_embeddings=None
    st.session_state.last_uploaded_file= uploaded_file

extracted_text=""

#Step 1: File Upload and Handling.
if uploaded_file is not None:
    st.write("File Uploaded Successfully")
    st.write("File Name:", uploaded_file.name)
    st.write("File Type:", uploaded_file.type)

    if st.session_state.faiss_index is None:
    #Step 2: Document Parsing and Text Extraction
        if uploaded_file.type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                doc=Document(uploaded_file)
                for paragraph in doc.paragraphs:
                    extracted_text += paragraph.text + "\n"
            
            except Exception as e:
                st.write(f"Error Reading Word Document: {e}")

        elif uploaded_file.type=="application/pdf":
            try:
                pdf_reader=PyPDF2.PdfReader(uploaded_file)
                for page_num in range(len(pdf_reader.pages)):
                    page=pdf_reader.pages[page_num]
                    extracted_text += page.extract_text() +"\n"

            except Exception as e:
                st.write(f"Error Reading PDF Document: {e}")

        else:
            st.warning("File Type not Supported")
            extracted_text=""

    #Step 3: Making chunks of the extracted Text
        if extracted_text:
            text_chunks= extracted_text.split("\n")
            text_chunks=[chunks.strip() for chunks in text_chunks if chunks.strip()] #It removes any extra spaces or blank lines at the beginning and end.It checks if, after removing the extra spaces, the segment still contains any actual text.If it does contain text, this cleaned-up version of the segment is added to a new text_chunks list.
            st.subheader("Extracted Text Chunks:")

            # This for is for displaying the chunks which is not needed
            #for i, chunks in enumerate(text_chunks):
            #    st.write(f"Chunks{i+1}:")
            #    st.info(chunks)
            
            st.write(f"Total No. of Chunks {len(text_chunks)}")

            embedding_model = st.session_state.embedding_model

            chunks_embeddings=[]
            st.subheader("Creating Embeddings")

            #progress_bar = st.progress(0)
            #status_text = st.empty()

            for i,chunk in enumerate(text_chunks):
                embedding=embedding_model.encode(chunk)
                chunks_embeddings.append(embedding)
            
            st.success("Embeddings Generated Successfully")
            st.session_state['chunks_embeddings']=chunks_embeddings
            st.session_state['text_chunks']=text_chunks

        else:
            st.warning("No text extracted to chunk")
            st.session_state.text_chunks=[]
            st.session_state.chunks_embeddings=[]

    #Step# 05: Building Faiss Index
        if st.session_state.text_chunks and st.session_state.chunks_embeddings:
                st.subheader("Building Faiss Index")
                embeddings_np= np.array(st.session_state['chunks_embeddings']).astype('float32')
                embedding_dim=embeddings_np.shape[1]
                faiss.normalize_L2(embeddings_np)
                index=faiss.IndexFlatIP(embedding_dim)
                index.add(embeddings_np)
                st.success("Faiss Build Sucessfully")
                st.write(f"No. of vectors in Faiss is {index.ntotal}")
                st.session_state['faiss_index']= index
        elif uploaded_file:
            st.warning("FAISS Index could not be built")

if st.session_state.faiss_index is not None and st.session_state.text_chunks:
    embedding_model=st.session_state.embedding_model
    faiss_index=st.session_state['faiss_index']
    llm_model=st.session_state.llm_model

    st.subheader("Chat with your document")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["parts"][0])

    if user_query:=st.chat_input("Enter Your Question About the Document"):
        st.session_state.messages.append({"role":"user","parts":[user_query]})

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                user_embedding= embedding_model.encode(user_query)
                user_embedding_np=np.array([user_embedding]).astype('float32')
                faiss.normalize_L2(user_embedding_np)
                distances,indices=faiss_index.search(user_embedding_np,k=1)
                #index.add(user_embeddings_np)

                most_similar_chunk_index = indices[0][0]
                most_relevant_chunk = st.session_state['text_chunks'][most_similar_chunk_index]

                chat=llm_model.start_chat(history=st.session_state.messages)

                llm_prompt_with_context=(
                    f"**Context:**\n{most_relevant_chunk}\n\n"
                    f"**User Question:**\n{user_query}\n\n"
                    f"Summarize the document if user ask for it"
                    f"Based ONLY on the provided context and uploaded document and the conversation history, answer the User Question. "
                    #f"If the answer cannot be found in the context or history, please state 'I cannot find the answer to this question in the provided document.' "
                    f"Do not introduce new information."
                )

                response=chat.send_message(llm_prompt_with_context, stream=True)

                full_answer = ""

                answer_placeholder = st.empty()

                for chunk in response:
                    if chunk.text is not None:
                        full_answer+= chunk.text
                        answer_placeholder.markdown(full_answer + "▌") 
                        time.sleep(0.02)
                answer_placeholder.success(full_answer)

            st.session_state.messages.append({"role":"model","parts":[full_answer]})
    
else:
    if uploaded_file is None:
        st.info("Please upload a document")

    else:
        st.warning("Document processing failed or no text extracted. Please check the file and try again.")


    

        
