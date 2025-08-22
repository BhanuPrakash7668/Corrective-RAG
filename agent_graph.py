from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, SkipValidation
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langchain_tavily import TavilySearch

from dotenv import load_dotenv

from prompts import GRADE_DOCUMENTS_PROMPT, QUESTION_REWRITER_PROMPT


KNOWLEDGE_BASE_URLS = [
    "https://devblogs.microsoft.com/cosmosdb/build-a-rag-application-with-langchain-and-local-llms-powered-by-ollama/?utm_source=chatgpt.com",
    "https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/",
]


class SharedState(TypedDict):
    """ Shared state for the RAG system """
    question: str
    agent_response: str
    vector_store: Chroma
    relevant_documents: list[str]
    model: ChatOpenAI
    
class GradeDocuments(BaseModel):
    """ Binary Score for relevant check on retrieved documents """
    
    binary_score: str = Field(
        desciption = "Documents are relevant to the question. 'Yes' or 'No'"
    )
    
def get_model(shared_state):
    shared_state['model'] = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)
    return shared_state

def build_vector_store(shared_state):
    """ Build a vector store from the knowledge base URLs """
    docs = [WebBaseLoader(url).load() for url in KNOWLEDGE_BASE_URLS]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    
    docs_split = text_splitter.split_documents(docs_list)
    
    vector_store = Chroma.from_documents(
        documents = docs_split,
        collection_name = "rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    shared_state['vector_store'] = vector_store
    
    return shared_state

def get_relevant_documents(shared_state):
    question = shared_state['question']
    vector_store = shared_state['vector_store']
    
    documents = vector_store.invoke(question)
    
    shared_state['relevant_documents'] = [doc.page_content for doc in documents]
    
    return shared_state


def grade_and_filter_documents(shared_state):
    
    print("\n\n Grading relevant documents...")
    
    question = shared_state['question']
    model = shared_state['model']
    documents = shared_state['relevant_documents']
    structured_llm_grader = model.with_structured_output(GradeDocuments)
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GRADE_DOCUMENTS_PROMPT),
            ("human", "Retrieved Documents: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    filtered_documents = []
    
    for document in documents:
        grader_response = retrieval_grader.invoke("question": question, "document": document)
        
        if grader_response.binary_score.lower() == "yes":
            print("---GRADE: Relevant ---")
            filtered_documents.append(document)
        else:
            print("---GRADE: Not Relevant ---")
    print("Filtered documents left after filtering:", len(filtered_documents))
    shared_state['relevant_documents'] = filtered_documents
    
    return shared_state


def generate_answer_from_documents(shared_state):
    model = shared_state['model']
    rag_prompt = hub.pull("rlm/rag-prompt")
    question = shared_state['question']
    documents = shared_state['relevant_documents']
    
    rag_chain = rag_prompt | model | StrOutputParser()

    model_response = rag_chain.invoke({"context": documents, "question": question})
    
    shared_state['agent_response'] = model_response
    
    return shared_state

def decide_to_generate(shared_state):
    
    if len(shared_state['relevant_documents']) > 0:
        print("\n Generating answer from relevant documents...")
        return "generate"
    else:
        print("\n No relevant documents found. Transform query and performing web search.....")
        return "transform_query"
    
    
def transform_query(shared_state):
    print("\n\n Transforming the question...")
    
    question = shared_state['question']
    model = shared_state['model']
    
    re_writer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUESTION_REWRITER_PROMPT),
            ("human", "Here is the inital question: {question} \n Formulate an important question."),
        ]
    )
    
    question_rewriter = re_writer_prompt | model | StrOutputParser()
    
    better_question = question_rewriter.invoke({"question": question})
    
    print('Transformed question:\n', better_question)           
    
    shared_state['question'] = better_question
    
    return shared_state


def perform_web_search(shared_state):
    print("\n\nPerforming web search...")
    
    question = shared_state['question']
    web_search_tool = TavilySearch()
    
    web_results = web_search_tool.invoke({"query": question})
    print('Web Search Results:\n', web_results['results'][0])
    
    documents  = [web_results['results'] for web_result in web_results['results']] 
    
    shared_state['relevant_documents'] = documents
    
    return shared_state



def build_graph(shared_state):
    workflow_graph = StateGraph(SharedState)
    
    workflow_graph.add_node("get_model", get_model)
    workflow_graph.add_node("build_vector_store", build_vector_store)
    workflow_graph.add_node("get_relevant_documents", get_relevant_documents)
    workflow_graph.add_node("grade_and_filter_documents", grade_and_filter_documents)
    workflow_graph.add_node("generate_answer_from_documents", generate_answer_from_documents)
    workflow_graph.add_node("perform_web_search", perform_web_search)
    wrokflow_graph.add_node("transform_query", transform_query)
    
    workflow_graph.add_edge(START, "get_model")
    workflow_graph.add_edge("get_model", "build_vector_store")
    workflow_graph.add_edge("build_vector_store", "get_relevant_documents")
    workflow_graph.add_edge("get_relevant_documents", "grade_and_filter_documents")
    workflow_graph.add_condtional_edges("grade_and_filter_documents",
                                 decide_to_generate,
                                 {
                                     "transform_query": "transform_query",
                                     "generate": "generate_answer_from_documents"
                                 },
                                )
    workflow_graph.add_edge("transform_query", "perform_web_search")
    workflow_graph.add_edge("perform_web_search", "generate_answer_from_documents")
    workflow_graph.add_edge("generate_answer_from_documents", END)
    
    
    return workflow_graph.compile()


