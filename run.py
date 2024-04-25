from langchain_core.prompts import  ChatPromptTemplate
f=open('api_key.txt')
GOOGLE_API_KEY=f.read()
from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",google_api_key=GOOGLE_API_KEY,temperature=0.2,convert_system_message_to_human=True)


from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate

chat_template = ChatPromptTemplate.from_messages([
        # System message prompt Template
        SystemMessagePromptTemplate.from_template("You are  a Helpful  AI data science  Tutor. Your name is {name}"),
        # Human Message
        HumanMessage(content="Hi!"),
        # AI Message
        AIMessage(content="HI there . How can I help you ?"),
        # Huma message Prompt Template
        HumanMessagePromptTemplate.from_template("{user_input}")
    

]

)
from langchain_core.output_parsers import StrOutputParser

output_parser= StrOutputParser()

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Leave No Context Behind.pdf")
pages = loader.load_and_split()


#Chunking
from langchain_text_splitters import NLTKTextSplitter

text_splitter = NLTKTextSplitter(chunk_size= 500,chunk_overlap=100)

chunks=text_splitter.split_documents(pages)

#print(len(chunks))

#print(type(chunks[0]))

#Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY,model="models/embedding-001")


from langchain_community.vectorstores import Chroma

db=Chroma.from_documents(chunks,embedding_model,persist_directory="./chroma_db_")
#persist the data base on the drive
db.persist()


#Setting a connection with ChromaDB

db_connection = Chroma(persist_directory="./chroma_db_",embedding_function=embedding_model)

#Converting Chroma db to retrieve object

retriever = db_connection.as_retriever(search_kwargs={"k":5})

#print(type(retriever))


from langchain_core.messages import SystemMessage

chat_template = ChatPromptTemplate.from_messages([
    # System Message from prompt Template
    SystemMessage(content='''You are a helpful AI bot
    You take the context and question from the user . Your answer should be  in a specific context .'''),
    
    #Human Message Prompt Template
    HumanMessagePromptTemplate.from_template('''Answer this question based on the given context.
    Context : {context}
    Question : {question}
    
    Answer : 
    ''' )
    



]
)

def getanswer(prompt):
    from langchain_core.runnables import RunnablePassthrough

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain =(
                {"context": retriever | format_docs,"question":RunnablePassthrough()}
                |chat_template
                |chat_model
                |output_parser

    )

    response = rag_chain.invoke(prompt)

    return response