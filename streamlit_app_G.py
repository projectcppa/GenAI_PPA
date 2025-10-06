#Chat bot using lang chain and Gemini
from itertools import zip_longest #helps combine two lists (like user messages and AI replies)
import streamlit as st #builds interactive web apps (UI).
from streamlit_chat import message #displays chat bubbles in the Streamlit UI
import langchain_google_genai #LangChain integration with Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI #wrapper to call Gemini via LangChain
#from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
) #schema objects that represent different roles in a chat (system = instructions, human = user, AI = bot)
#openapi_key= st.secrets["OPENAI_API_KEY"]

# Set streamlit page configurationsst.set_page_config(page_title="Hope to Skill ChatBot")
st.set_page_config(page_title="Hope to Skill ChatBot")
st.title("Alina GPT")  #Setting Streamlit page title

# Initialize session state variables
#generated: AI responses list, past: user inputs list, entered_prompt: latest submitted question from user.
if 'generated' not in st.session_state:
    st.session_state['generated']=[] #Store AI generated Response so that chat history doesn’t reset on each interaction 

if 'past' not in st.session_state:
    st.session_state['past']=[]   #Store past user inputs

    if 'entered_prompt' not in st.session_state:
        st.session_state['entered_prompt']="" 

# Initialize the ChatOpenAI model
#google.generativeai.configure(api_key="AIzaSyB_OHmnzesSf4O1FeM_9PDFim9N9wPmZqg") #This line creates an instance of the Client class from the google.genai library.The Client object is your primary interface for sending requests to and receiving responses from the Gemini models.

#how Gemini generates answers
generation_config= {
    "temperature":0,
    "top_p":0.95,
    "top_k":40,
    "max_output_tokens":133,
    "response_mime_type":"text/plain", #mime handling data types
    }
chat= ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyB_OHmnzesSf4O1FeM_9PDFim9N9wPmZqg",model_kwargs=generation_config)
                            # generation_kwargs=generation_config)

def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    #here system message is setting rules for starting conv
    zipped_messages=[SystemMessage(
        
       content= """your name is AI mentor. You are an AI Technical Expert for Artificial Intelligence. Ask user about their Name before starting and you are here to guide and assist students with their AI-related queries.           

            1. Greet the user politely. ask user name and ask how can you assist with AI-related queries.
            2. Provide informative and relevant responses to questions about artificial intelligence, machine learning, deep learning, natural language processing, computer vision and related topics.
            3. You must avoid discussing sensitive, offensive and harmful content. Refrain from engaging in any for or discrimination, harrasement or inappropriate behavior.
            4. If user ask about topic unrelated to AI, politely steer the conversation back to AI or Inform them that the topic is outside the scope of this conversation.
            5. Be patient and considerate when responsing to the user queries, and provide clear explanations.
            6. If the user expresses gratitude or indicates end of the conversation, respond with a polite farewell.
            7. Do not generate long paragraphs in response. Maximum should be 100

            Remember, your primary goal is to assist and educate students in the field of Artificial Intelligence. Always prioritze their learning experience and well-being. """
    )]


    # Zip together the past and generated messages and Returns a complete conversation list that gets sent to Gemini
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg)) #Add user messages
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg))     #Add AI Messages
        
    return zipped_messages



#Builds conversation history, sends to gemini and Returns only the text (content) of AI response.
def generate_response():
    """
    Generate AI response using ChatOpenAI model.
    """
    # Build the list of messages
    zipped_messages= build_message_list()
    
    #Generate response using the chat model
    ai_response=chat(zipped_messages)

    return ai_response.content


# Define function to submit user input
#Runs when user submits input, Stores the current text in entered_prompt, Clears input box (prompt_input). 
def submit():
    #set entered prompt to the current value of prompt_input
    st.session_state.entered_prompt= st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input=""


# Create text input for user
st.text_input('YOU:', key='prompt_input', on_change=submit)


if st.session_state.entered_prompt !="":  
    # Get user query
    user_query= st.session_state.entered_prompt

    #append user query to past queries
    st.session_state.past.append(user_query)

    # Generate Response
    output= generate_response()

    # Append AI response to generated responses
    st.session_state.generated.append(output)

# Display the chat history
#Loops backwards through chat history (latest message on top)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1,-1,-1):
        # Display AI Response
        message(st.session_state["generated"][i], key=str(i))
        # Display user message
        message(st.session_state["past"][i],
                is_user=True, key=str(i) + '_user')


    