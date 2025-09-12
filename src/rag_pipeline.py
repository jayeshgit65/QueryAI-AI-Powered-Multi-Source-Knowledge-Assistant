from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Create a context-aware retriever chain
def get_context_retriever_chain(vector_store):
    llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up information relevant to the conversation.")
    ])

    return create_history_aware_retriever(llm, retriever, prompt)

# Build a Retrieval-Augmented Generation (RAG) conversational chain
def get_conversational_rag_chain(retriever_chain):
    llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context below:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_chain)

# Generate a response to user input using RAG pipeline
def get_response(user_input, vector_store, chat_history):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    return response["answer"]
