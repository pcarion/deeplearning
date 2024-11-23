from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# we load the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

REDIS_URL = "redis://localhost:6379"


# we load the markdown file
markdown_path = "./us-constitution.md"
loader = UnstructuredMarkdownLoader(markdown_path)
documents = loader.load()

# print number of documents
print("number of documents: ", len(documents))

# we print the length of the first document's page content
print("length of the first document's page content: ", len(documents[0].page_content))

# we split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(documents)

print("number of splits: ", len(all_splits))


config = RedisConfig(
    index_name="newsgroups",
    redis_url=REDIS_URL,
    metadata_schema=[
        {"name": "category", "type": "tag"},
    ],
)

vectorstore = RedisVectorStore.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())


print("number of vectors: ", vectorstore)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# retrieved_docs = retriever.invoke("Can a president be elected for a third term if the last term was was more than 4 years ago?")

# print("number of retrieved documents: ", len(retrieved_docs))

# # loop through the retrieved documents and print the page content
# for doc in retrieved_docs:
#     print("-" * 80)
#     print("metadata of the document: ", doc.metadata)
#     print("content of the document: ", doc.page_content)
#     print("-" * 80)
#     print(doc)
#     print("-" * 80)

prompt = hub.pull("rlm/rag-prompt")

print("prompt:")
print("-" * 80)
print(prompt)
print("-" * 80)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("Can a president be elected for a third term if the last term was was more than 4 years ago?"):
    print(chunk, end="", flush=True)
