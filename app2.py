# Import necessary libraries
import os, re
from flask import Flask, render_template, request, redirect
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
#import shutil
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA, LLMChain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tempfile import NamedTemporaryFile

# Get the OpenAI API key from the environment variable
api_key = "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"

if api_key is None or api_key == "":
    print("OpenAI API key not set or empty. Please set the environment variable.")
    exit()  # Terminate the program if the API key is not set.

# Initialize the OpenAI client with the API key
os.environ['OPENAI_API_KEY'] = api_key
FAISS_PATH = "/faiss"

# Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []

def get_document_loader():
    loader = DirectoryLoader('documents', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_embeddings():
    documents = get_document_loader()
    chunks = get_text_chunks(documents)
    db = FAISS.from_documents(
        chunks, OpenAIEmbeddings()
    )
    return db


def get_retriever():
    db = get_embeddings()
    retriever = db.as_retriever()
    return retriever

def get_claim_approval_context():
    db = get_embeddings()
    context = db.similarity_search("What are the documents required for claim approval?")
    claim_approval_context = ""
    for x in context:
        claim_approval_context += x.page_content

    return claim_approval_context

def get_general_exclusion_context():
    db = get_embeddings()
    context = db.similarity_search("give me list of all general exclusions")
    general_exclusion_context = ""
    for x in context:
        general_exclusion_context += x.page_content

    return general_exclusion_context

def get_file_content(file):
    text = ""
    if file.filename.endswith(".pdf"):
        pdf = PdfReader(file)
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text += page.extract_text()

    return text

pdf_llm = ChatOpenAI(model="gpt-3.5-turbo")



PROMPT = """You are an AI assistant for verifying health insurance claims. You are given with the references for approving the claim and the patient details. Analyse the given data and predict if the claim should be accepted or not. You the following guidelines for your analysis.

1.Verify if the patient has provided all necessary information and all necessary documents
and if you find any incomplete information or required documents are not provided then set INFORMATION criteria as FALSE. 
if patient has provided all required documents then set INFORMATION criteria as TRUE. 

2. In any disease of the patient is in the general exclusions list, set EXCLUSION criteria as FALSE otherwise set it to TRUE
Use this information to verify if the application is valid and to accept or reject the application.

DOCUMENTS FOR CLAIM APPROVAL: {claim_approval_context}

MAKE SURE TO CHECK 
go through the document {patient_info} and find what is the total claim amount? Your job is to accept or reject the claim application. Our claim limit is Total Claim Amount. If total claim amount is more than claim limit reject the claim else accept the claim.
    

IF THE CLAIM AMOUNT IS LESS THAN THE TOTAL CLAIM LIMIT THEN ACCEPT THE CLAIM OTHERWISE REJECT THE CLAIM.
    PATIENT INFO : {patient_info}

 


Use the above information to verify if the application is valid and decide if the application has to be accepted or rejected keeping the guidelines into consideration. 

Generate a detailed report about the claim and procedures you followed for accepting or rejecting the claim and the write the information you used for creating the report. 
Create a report in the following format

write whether INFORMATION AND EXCLUSION List are true or false 
reject the claim if any of them is false.
write whether claim is accepted or not.

Executive Summary
[Provide a Summary of the report.]

Introduction
[Write a paragraph about the aim of this report, and the state of the approval.]

Claim Details
[Provide details about the submitted claim]

Claim Description
[Write a short description about claim]

Document Verification
[Mentions which documents are submitted and if they are verified.] 

Document Summary
[Give a summary of everything here including the medical reports of the patient]

Please verify for any signs of fraud in the submitted claim if you find the documents required for accepting the claim for the medical treatment.
"""


def check_claim_rejection(claim_reason, general_exclusion_list, prompt_template,patient_info,threshold=0.5):
    vectorizer = CountVectorizer()
    patient_info_vector = vectorizer.fit_transform([claim_reason])
    
    for disease in general_exclusion_list:
        disease_vector = vectorizer.transform([disease])
        similarity = cosine_similarity(patient_info_vector, disease_vector)[0][0]
        if similarity > threshold:
            
            prompt_template = f"""You are an AI assistant for verifying health insurance claims. You are given with the references for approving the claim and the patient details. Analyse the given data and give a good rejection. You the following guidelines for your analysis.
            make sure that the claim amount is less than the CLAIM LIMIT then accept the claim otherwise reject the claim.
            PATIENT INFO : {patient_info}

            Executive Summary
                [Provide a Summary of the report.]

                Introduction
                [Write a paragraph about the aim of this report, and the state of the approval.]

                Claim Details
                [Provide details about the submitted claim]

                Claim Description
                [Write a short description about claim]

                Document Verification
                [Mentions which documents are submitted and if they are verified.] 

                Document Summary
                [Give a summary of everything here including the medical reports of the patient]
            
            CLAIM MUST BE REJECTED: Patient has {disease}."""
            return prompt_template
    
    return prompt_template



def check_claim_amt(medical_bill):
    pdf_loader = PyPDFLoader(medical_bill)
    docs = pdf_loader.load()
    pdf_chunks = get_text_chunks(docs)
    pdf_db = FAISS.from_documents(pdf_chunks, OpenAIEmbeddings())
    retriever = pdf_db.as_retriever()
    context = pdf_db.similarity_search("claim amount")
    PDF_PROMPT = """go through the document {context} and find what is the claim amount? """
    pdf_prompt_template = PromptTemplate(
        template=PDF_PROMPT,
        input_variables=["context"]
    )
    llmchain = LLMChain(llm=pdf_llm, prompt= pdf_prompt_template)
    return llmchain.invoke({"context": context})['text']






general_exclusion_list = ["HIV/AIDS", "Parkinson's disease", "Alzheimer's disease","pregnancy", "substance abuse", "self-inflicted injuries", "sexually transmitted diseases(std)", "pre-existing conditions"]





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def msg():
    name = request.form['name']
    address = request.form['address']
    claim_type = request.form['claim_type']
    claim_reason = request.form['claim_reason']
    date = request.form['date']
    medical_facility = request.form['medical_facility']
    medical_provider = request.form['medical_provider']
    medical_bill = request.files['medical_bill']
    total_claim_amount = request.form['total_claim_amount']
    total_expenses = request.form['total_expenses']
    description = request.form['description']

    bill = get_file_content(medical_bill)
    print(bill)
    
    patient_info = f"Name: {name} " + f"\nAddress: {address} " + f"\nClaim type: {claim_type} " + f"\nClaim reason: {claim_reason}" + f"\nMedical facility: {medical_facility} " + f"\nMedical bill: {bill}" + f"\nDate : {date} " + f"\nMedical provider: {medical_provider}" + f"\nTotal claim amount: {total_claim_amount}" + f"\nTotal expenses: {total_expenses}" + f"\nDescription: {description}"


    claim_reason = f'Claim Reason: {claim_reason}'
    #print(patient_info)
    print(claim_reason)
    # prompt_template = "Please provide details of the insurance claim."
    
    updated_prompt = check_claim_rejection(claim_reason, general_exclusion_list,patient_info,PROMPT)
    # print(updated_prompt)
    prompt = PromptTemplate(input_variables=["claim_approval_context","patient_info","claim_amt"], template=updated_prompt)
    print(prompt)
   
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    llmchain = LLMChain(llm=llm, prompt= prompt)
    output = llmchain.run({"claim_approval_context": get_claim_approval_context(), "patient_info": patient_info})
    
    output = re.sub(r'\n', '<br>', output)
    
    return render_template("result.html", name=name, address=address, claim_type=claim_type, claim_reason=claim_reason, date=date, medical_facility=medical_facility, medical_provider=medical_provider, total_claim_amount=total_claim_amount, total_expenses=total_expenses, description=description, output=output)

if __name__ == '__main__':
    app.run(debug=False) 