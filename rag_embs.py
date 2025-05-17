import streamlit as st
from openai import OpenAI
from st_social_media_links import SocialMediaIcons

from .utils.prompt import get_query_results, prepare_prompt

## OpenAI stuff
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])
# set up the model and load chat history
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4.1-mini"
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar.container(border=False):
    st.sidebar.markdown(
        """
    # LLM with RAG
    - [Description](#desc)
    - [Chat time!](#chat)
    - [More details](#info)
    """,
        unsafe_allow_html=True,
    )

with st.sidebar.container(border=False):
    st.sidebar.markdown("""
    # About me
    You can contact me on one of the channels:
    """)
with st.sidebar.container(border=False):
    linkedin_image = "https://media.licdn.com/dms/image/v2/C4E03AQEYVPd4oVI9xg/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1653659430741?e=1750291200&v=beta&t=_scrO_7DxXjfi4VfOBsZ1GWpYwE00b4wwZDrd9PkMvo"
    _, _, cent, _, _ = st.columns(5)
    with cent:
        st.image(linkedin_image, width=100)
with st.sidebar.container(border=False):
    social_media_links = [
        "https://www.linkedin.com/in/rpcarvs/",
        "mailto:rodrigo.carvalho.al@gmail.com",
    ]
    social_media_icons = SocialMediaIcons(
        social_media_links,
        colors=["#0A66C2", "#EF4026"],
    )
    social_media_icons.render()

with st.container(border=False):
    st.title("Example of LLM with RAG")
    st.subheader("Description", divider=True, anchor="desc")
    st.markdown(
        """<div style="text-align: justify;">This page shows a simple implementation of Retrieval-Augmented
Generation (RAG). This approach enables a better contextualization
when interacting with LLM chatbots. 
        
A very short summary of the steps involved:
        
1. I have built a small "context library" containing a set of papers 
and reviews on NLP, text embeddings, and LLMs.
2. The library fueled a vector database with thousands of quotes from 
all its documents. The database is stored in [Atlas/MongoDB](https://www.mongodb.com/products/platform/atlas-database)
using the Microsoft Azure infrastructre.
3. The quotes were mapped into a 384-dimensional vector using the 
open-source [all-MiniLM-L6-v2 model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), thus enabling semantic search.
4. Part of the prompt is used to generate a better context, gathering
info from the vector database through semantic search.
5. The chat prompt is prepared with the context to be interfaced with
OpenAI ChatGPT.</div>
""",
        unsafe_allow_html=True,
    )

with st.container(border=False):
    st.subheader("Chatting with OpenAI", divider=True, anchor="chat")

    st.markdown(
        """<div style="text-align: justify;">
Use the 'Context for RAG' field to add the relevant context
for the semantic search. This text will be converted to embeddings and used in
a [HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/) approach to search
for similar quotes in our knowledge-base vector database. After this, ask your
questions in the chat field. Try to focus on text embeddings as it is the
only topic in our library.</div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    # get query text for RAG
    rag_context = st.text_input(
        "Context for RAG",
        value="Accuracy of open-source embedding models",
    )

    st.markdown("Chat here:")
    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me about vector embeddings..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # get enhanced prompt
        if rag_context:
            documents = get_query_results(
                rag_context,
                username=st.secrets.db_credentials.username,
                password=st.secrets.db_credentials.password,
                limit=5,
            )

            prompt = prepare_prompt(prompt, documents)

        # Add user message to chat history after context
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})


with st.container(border=False):
    st.subheader("More details", divider=True, anchor="info")

    st.markdown(
        """<div style="text-align: justify;">
In Retrieval-Augmented Generation (RAG) techniques, information from external 
sources can enhance interaction with LLM chatbots. For instance, leveraging a 
company's existing knowledge base (commonly stored in a database) provides better 
context for the chatbot. This approach improves the quality of the LLM's responses 
by ensuring the prompt is processed with a richer context. The figure below 
illustrates this process, outlining every significant step in the workflow.
</div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    st.image(
        image="./imgs/flowchart.png",
        caption="Illustrating how a RAG workflow can look like.",
        use_container_width=True,
    )
    st.write("")

    st.markdown(
        """<div style="text-align: justify;">
Text embedding models are limited by their training data and model architecture, 
making it challenging to process lengthy documents containing multiple pages. 
To address this limitation, an efficient approach is to split the document into 
smaller parts, known as chunks. Text is extracted from these chunks and used to 
generate model embeddings, which are then stored in a vector database. The 
figure below illustrates this workflow.
</div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    st.image(
        image="./imgs/drawing.png",
        caption="Brief description of the chunking and embedding processes.",
        use_container_width=True,
    )
