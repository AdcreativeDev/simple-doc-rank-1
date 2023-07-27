# import yaml

import streamlit as st
import os
from langchain import OpenAI, VectorDBQA, LLMChain
from langchain.prompts import PromptTemplate
from pdf_loaders import PdfToTextLoader
from dataset_vectorizers import DatasetVectorizer
from PIL import Image

st.set_page_config(
    page_title="Document Extract and Ranking",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)



# with open("config.yml", "r") as f:
#     config = yaml.safe_load(f)
# OPENAI_API_KEY = config['OPENAI_KEY']

PDFS, NAMES, TXTS = [], [], []
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500

# ----- Header of the app -----
st.title("🔎 📄 + 📄 -> ⛓ -> 📋 :blue[Documents Extraction and Evaluation with ranking]")
st.subheader('Sample Use case: Product or Services comparison')
st.write('Job resume screening, product features comparison, Insurance Policy comparison ... ')

col1, col2, col3 = st.columns(3)
with col1:
    system_openai_api_key = os.environ.get('OPENAI_API_KEY')
    system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
    os.environ["OPENAI_API_KEY"] = system_openai_api_key

with col2:
    st.write('Sample screenshot')
    with st.expander('Click here'):
        image = Image.open('doc-step-1.jpg')
        st.image(image)
        image = Image.open('doc-step-2.jpg')
        st.image(image)


# ----- Select and upload the files one by one -----
st.subheader("📤 :blue[Step 1 - Upload 2 PDF files for comparsion]")
col1, col2 = st.columns(2)
with col1:
    file_1 = st.file_uploader("📄 Upload NO.1 PDF")
    name_1 = st.text_input("Enter Name of file 1", value="Plan 1")

with col2:
    file_2 = st.file_uploader("📄Upload NO.2 PDF")
    name_2 = st.text_input("Enter Name of file 2", value="Plan 2")

# ----- Load the files -----
if file_1 and file_2:

    with open("./data/" + file_1.name, "wb") as f:
        f.write(file_1.getbuffer())

    with open("./data/" + file_2.name, "wb") as f:
        f.write(file_2.getbuffer())

    PDFS = ["./data/" + file_1.name, "./data/" + file_2.name]
    NAMES = [name_1, name_2]

    for pdf_path in PDFS:
        txt_path = pdf_path.replace(".pdf", ".txt")
        pdf_loader = PdfToTextLoader(pdf_path, txt_path)
        text = pdf_loader.load_pdf()
        TXTS.append(txt_path)

    # llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=system_openai_api_key)

    with st.spinner('Read File and Vectorization'):    
        st.write("✔️  PDF Files loaded.")
        llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=OPENAI_API_KEY)
        st.write("✔️  Connect to OpenAi.")
        dataset_vectorizer = DatasetVectorizer()
        documents_1, texts_1, docsearch_1 = dataset_vectorizer.vectorize([TXTS[0]], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, openai_key=OPENAI_API_KEY)
        documents_2, texts_2, docsearch_2 = dataset_vectorizer.vectorize([TXTS[1]], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, openai_key=OPENAI_API_KEY)
        st.write("✔️ 2 Chroma VectorStore created.")
        qa_chain_1 = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch_1)
        qa_chain_2 = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch_2)
        st.write("✔️ 2 Langchain DB Chain Created.")

    # ----- Write questions separated by a new line -----
    st.subheader(":blue[Step 2 - Create questions to generate a summary]")
    col1, col2, col3 = st.columns(3)
    with col1:    
        company_description = st.text_area("Brief company description", value="full-remote consulting company with 100 employees")

    with col2:
        st.write("🔎 :green[Extraction Prompt]")
        st.write("⬇️  Content extract rules:)")
        questions = st.text_area("Questions", 
                                value = """
                                How good are the deductibles?
                                How is the preventive care coverage?
                                How this plan fits for remote workers in the US and abroad?
                                What is the maximum money amount that can be compensated?
                                Can I go to any hospital of my choice?
                                Are there any limitations that won\'t allow to use the insurance?
                                Does it cover the family members of the applicant?
                                What are the healthcare procedures that are not covered by the insurance?
                                Can I use the insurance for the dental care?
                                Can I use the insurance in other countries?""")


    with col3:
        # ----- Select final criteria for decision-making -----
        st.write("🔎 :green[Ranking Search Result Prompt]")
        st.write("⬇️  Evaluated and Ranking rules:")
        criteria = st.text_area("Criteria", value="""
                    1. Coverage of different health procedures
                    2. Flexibility for remote workers abroad
                    3. Price and compensation""")
                                

    st.subheader(':blue[Step 3 - Press [START] - extraction &  comparison]')     
    if st.button('START', type='primary'):
        with st.spinner('Processing ....'):     

            QUESTIONS = questions.split("\n")
            QUESTIONS = [q.strip() for q in QUESTIONS if len(q) > 0]

            CRITERIA = criteria.split("\n")
            CRITERIA = [c.strip() for c in CRITERIA if len(c) > 0]
            final_criteria = "".join([f"{i}. {c}\n" for i, c in enumerate(CRITERIA, 1)])


            col1 , col2 = st.columns(2)
            with col1:

                st.write("✔️ Preparing extraction and ranking statement")
                # ----- Generate the intermediate answers for the document summary -----
                summary_of_answers = ""
                nquery = 1
                st.write(f'✔️ Start Extraction')
                for q in QUESTIONS:
                    # print(q)
                    answer_1, answer_2 = qa_chain_1.run(q), qa_chain_2.run(q)
                    st.write(f'✔️ Extracted Q: {str(nquery)} : {str(q[:20])} ..., answer_1: {answer_1[:10]} ..., answer_2: {answer_2[:10]} ...')
                    summary_of_answers += "Question: " + q + "\n"
                    summary_of_answers += f"{NAMES[0]} answer: " + answer_1 + f";\n {NAMES[1]} answer: " + answer_2 + "\n"
                    nquery += 1
                
                st.write(f'✔️ Total Length of answer : {str(len(summary_of_answers))}')

                    
                template = """
                    I want you to act as an expert in insurance policies. 
                    I have asked two companies about their insurance policies and here are their answers:
                    {summary_of_answers}
                    I am looking for insurance for a {company_description}. 
                    I want you to tell me which company is better and why.
                    Give me a rating (x out of 10) for the following categories for each company separately with a short explanation (50 words max) for each category:
                    {final_criteria}
                    Your answer and final recommendation after the rating:
                    """
                
            
            with col2:
                st.subheader(':blue[Step 4 - LangChain Prompt Execution]')                            

                prompt = PromptTemplate(
                    input_variables=["summary_of_answers", "company_description", "final_criteria"],
                    template=template,
                )
                st.write(f'✔️ Prompt Template create and pass to Langchain ')
                
                answer = ""
                llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=OPENAI_API_KEY, request_timeout=60)
                st.write(f'✔️ Start execute the Chain ')

                chain = LLMChain(llm=llm, prompt=prompt)
                answer = chain.run({"summary_of_answers": summary_of_answers, "final_criteria": final_criteria, "company_description": company_description})
                st.write(f'✔️ Completed execute the Chain ')


                # ----- Generate the final answer -----
                st.subheader("✅  :orange[Step 5 : Evaluation Result]")
                st.info(answer)



