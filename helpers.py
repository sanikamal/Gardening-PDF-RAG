import os
import glob
import shutil
from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# if os.getenv("GITHUB_TOKEN"):
# 	print("GITHUB TOKEN found!")

PDF_DOCS_PATH = "./docs"
EMBEDDINGS = OpenAIEmbeddings(api_key=os.getenv("GITHUB_TOKEN"), base_url=os.getenv("ENDPOINT"), model = "text-embedding-3-large")

# print(pdf_files)
def clean_text(text: str) -> str:
    # Replace '\n' with spaces and remove multiple spaces
    cleaned = " ".join(text.split("\n"))  # Join on newlines first
    cleaned = " ".join(cleaned.split())  # Remove extra spaces
    return cleaned

def load_pdf(pdf_path):
	# Use the PyPDFLoader by specifying the correct path
	loader = PyPDFLoader(pdf_path)
	# Use the load method from the loader to get the documents
	documents = loader.load()
	# Iterate over the documents
	for document in documents:
		# Apply the clean_text function to the page_content attribute of each document
		document.page_content = clean_text(document.page_content)
	return documents

def split_docs(docs):
	# Instantiate the RecursiveCharacterTextSplitter class with the appropriate parameters
	text_splitter = RecursiveCharacterTextSplitter( 
		chunk_size=1000,
		chunk_overlap=100,
		separators=[". ", "? ", "! ", "\n\n", "\n", " ", ""], 
		keep_separator=True,
	)

	# Use the splitter to split the documents (use the split_documents method)
	splits = text_splitter.split_documents(docs)

	return splits

def setup_vectordb(splitted_docs, db_docs_path="db/chroma/"):

	# Delete the in-memory directory that will hold the data
	# This is done in case you run this function multiple times to avoid duplicated documents
	if os.path.exists(db_docs_path) and os.path.isdir(db_docs_path):
		shutil.rmtree(db_docs_path)

	# Create an instance of the vector database
	vectordb = Chroma.from_documents( 
		documents=splitted_docs,
		embedding=EMBEDDINGS,
		persist_directory=db_docs_path,
	)

	return vectordb


# # Get all .pdf files in the base directory and its subdirectories
# pdf_files = glob.glob(os.path.join(PDF_DOCS_PATH, "*.pdf"))
# docs = [doc for pdf in pdf_files for doc in load_pdf(pdf)]
# # print(f"There are a total of {len(docs)} documents")
# splitted_docs = split_docs(docs)
# # print(f"There are a total of {len(splitted_docs)} documents after splitting")
# DATABASE = setup_vectordb(splitted_docs)

# if os.path.exists("./db/chroma/"):
# 	print("Successfully created the vector database!")
# else:
# 	print("The directory to store the vector database was not created, double check your code.")
	
# question = "How can I plant tomatoes?"
# retrieved_docs = DATABASE.similarity_search(question, k=5)

# for rd in retrieved_docs:
# 	print(rd)

def get_retriever(persist_directory="db/chroma/"):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=EMBEDDINGS)
    retriever = vectordb.as_retriever()
    return retriever

def format_docs(docs):
    results = []
    for i, doc in enumerate(docs, 1):
        source_path = doc.metadata.get("source", "Unknown")
        filename = (
            os.path.basename(source_path) if source_path != "Unknown" else "Unknown"
        )

        result = (
            f"Result {i}:\n"
            f"{doc.page_content}\n\n"
            f"Document: {filename}\n"
            f"Page: {doc.metadata.get('page', 'Unknown')}"
        )
        results.append(result)

    return "\n---\n".join(results)

# print(format_docs(retrieved_docs))

def process_query(question):
	# Initialize the LLM
	llm = ChatOpenAI(api_key=os.getenv("GITHUB_TOKEN"), base_url=os.getenv("ENDPOINT"), model_name="gpt-4o", temperature=0)

	# Define a template for the QA
	template = """Use the following pieces of context to answer the question at the end. Provide a detailed, thorough response that:
		1. Answers the main question
		2. Provides relevant examples or details from the context
		3. Explains any important concepts mentioned
		4. If relevant, discusses implications or applications

		If you don't know the answer, provide a detailed explanation of what aspects you're uncertain about and why.

		{context}
		Question: {question}
		Detailed Answer:"""

	# Instantiate a PromptTemplate using the template given
	prompt = PromptTemplate.from_template(template)

	# Use the as_retriever method to use the DATABASE as a retriever
	retriever = get_retriever()
	
	# Get the source documents by using the invoke method on the retriever and passing the question
	source_documents = retriever.invoke(question)

	# Format the source documents using the format_docs helper function
	doc_references = format_docs(source_documents)

	# Set up the QA chain
	qa_chain = ( 
	# Use the retriever as context and a RunnablePassthrough as question
	{"context": retriever, "question": RunnablePassthrough()}
	# Pipe to the prompt
	| prompt
	# Pipe to the llm
	| llm
	# Pipe to the StrOutputParser
	| StrOutputParser()
	) 

	# Get response from qa_chain by using the invoke method and passing the question
	llm_response = qa_chain.invoke(question)
	return llm_response, doc_references

# question = "How can I plant flowers?"

# llm_response, doc_references = process_query(question)

# print(f"### LLM Response #################\n\n{llm_response}\n")
# print(f"### References ###################\n\n{doc_references}")