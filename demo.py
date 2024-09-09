import streamlit as st
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import os
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import KnowledgeGraphIndex, ServiceContext
import time

os.environ["OPENAI_API_KEY"] = "YOUR KEY"
# os.environ["NEBULA_USER"] = "root"
# os.environ["NEBULA_PASSWORD"] = "nebula"  # default is "nebula"
# os.environ[
#     "NEBULA_ADDRESS"
# ] = "127.0.0.1:9669"  # assumed we have NebulaGraph installed locally

@st.cache_data
def set_llm(model_name):
    Settings.llm = OpenAI(
        model=model_name
    )

@st.cache_resource
def get_query_engine(file, persist, index_type):
    if file == "essay":
        if index_type == "vector":
            if not os.path.exists(persist):
                document = SimpleDirectoryReader(input_files=["./files/essay.txt"]).load_data()
                index = VectorStoreIndex.from_documents(document)
                index.storage_context.persist(persist_dir=persist)
                print("builing persisit for essay")
            else:
                storage_context = StorageContext.from_defaults(persist_dir=persist)
                index = load_index_from_storage(storage_context)
                print("loading persist for essay")
        query_engine = index.as_query_engine()
    else:
        if index_type == "vector":
            if not os.path.exists(persist):
                reader = SimpleWebPageReader(html_to_text=True)
                doc = reader.load_data(urls=["https://en.wikipedia.org/wiki/Guardians_of_the_Galaxy_Vol._3"])
                index = VectorStoreIndex.from_documents(doc)
                index.storage_context.persist(persist_dir=persist)
                print("building persist for wiki")
            else:
                storage_context = StorageContext.from_defaults(persist_dir=persist)
                index = load_index_from_storage(storage_context)
                print("loading persist for wiki")
            query_engine = index.as_query_engine()
        # else:
        #     graph_store = NebulaGraphStore(
        #         space_name="ggwiki3",
        #         edge_types=['relationship'],
        #         rel_prop_names=['relationship'],
        #         tags=['entity']
        #     )
        #     storage_context = StorageContext.from_defaults(graph_store=graph_store)
        #     kg_index = KnowledgeGraphIndex(nodes=[], storage_context=storage_context)
        #     query_engine = kg_index.as_query_engine()
        #     print("loading from nebula database")
    return query_engine

def getPrompt(input):
    prompt = f"from the given data source, respond this query: {input}"
    return prompt

def stream_data(answer):
    for word in answer.split(" "):
        yield word + " "
        time.sleep(0.02)

model_name = "gpt-4o"
set_llm(model_name)

custom_file = st.sidebar.selectbox("custom_file", options=["essay", "website"], index=0)
persist_essay = "./PERSIST_ESSAY"
persist_website = "./PERSIST_WIKI"
if custom_file == "essay":
    persist = persist_essay
else:
    persist = persist_website
index_type = st.sidebar.selectbox("Index type", options=["vector"], index=0)

query_engine = get_query_engine(file=custom_file, persist=persist, index_type=index_type)

if st.sidebar.button("clear"):
    st.session_state.clear()
    set_llm.clear()
    get_query_engine.clear()
    st.cache_data.clear()
    st.cache_resource.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("RAG CHATBOT")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
if query := st.chat_input("input query:"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    prompt = getPrompt(query)
    
    with st.chat_message("assistant"):
        answer = query_engine.query(prompt)
        print(answer.get_formatted_sources())
        response = st.write_stream(stream_data(answer.response))
    st.session_state.messages.append({"role": "assistant", "content": response})
