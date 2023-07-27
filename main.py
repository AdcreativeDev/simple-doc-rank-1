import streamlit as st
import yaml
import tiktoken

from langchain import OpenAI, VectorDBQA, LLMChain
from langchain.prompts import PromptTemplate

from pdf_loaders import PdfToTextLoader
from dataset_vectorizers import DatasetVectorizer

if st.button('start'):
    with st.spinner('working...'):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)

        OPENAI_API_KEY = config['OPENAI_KEY']

        st.write(OPENAI_API_KEY)

        # data taken from https://www.freelancersunion.org/insurance/health/bronze-plans-nyc/
        # 3 lists 
        PDFS, NAMES, TXTS  = [
            './data/insurance-policy1.pdf',
            './data/insurance-policy2.pdf'
        ], [
            'COMPANY 1', 
            'COMPANY 2'
        ], []


        for pdf_path in PDFS:
            # create a txt filename from pdd file
            txt_path = pdf_path.replace(".pdf", ".txt")
            # The PdfToTextLoader class is initialized with the pdf_path and txt_path as arguments, representing the input PDF file path and the output text file pat
            pdf_loader = PdfToTextLoader(pdf_path, txt_path)
            # pdf_loader reads the PDF file specified by pdf_path ,  text is a variable that holds the extracted text content from the PDF file.
            text = pdf_loader.load_pdf()
            # the path to the generated text file) is appended to the TXTS list.
            TXTS.append(txt_path)


        st.write(len(TXTS))
        st.write(PDFS[0])
        st.write(PDFS[1])
        st.write(TXTS[0])
        st.write(TXTS[1])
        st.write(NAMES[0])
        st.write(NAMES[1])

        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 500

        dataset_vectorizer = DatasetVectorizer()
        documents_1, texts_1, docsearch_1 = dataset_vectorizer.vectorize([TXTS[0]], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, openai_key=OPENAI_API_KEY)
        documents_2, texts_2, docsearch_2 = dataset_vectorizer.vectorize([TXTS[1]], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, openai_key=OPENAI_API_KEY)


        QUESTIONS = [
            'How good are the deductibles?',
            "How is the preventive care coverage?",
            'How this plan fits for remote workers in the US and abroad?',
            'What is the maximum money amount that can be compensated?',
            'Can I go to any hospital of my choice?',
            'Are there any limitations that won\'t allow to use the insurance?',
            'Does it cover the family members of the applicant?',
            'What are the healthcare procedures that are not covered by the insurance?',
            'Can I use the insurance for the dental care?',
            'Can I use the insurance in other countries?'
        ]

        st.write(QUESTIONS[0])
        st.write(QUESTIONS[1])



        llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=OPENAI_API_KEY)
        # VectorDBQA provides question-answering functionality based on vectorized text data.
        qa_chain_1 = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch_1)
        qa_chain_2 = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch_2)

        summary_of_answers = ""
        # This is a loop that iterates over each question in the list QUESTION
        for q in QUESTIONS:
            st.write(q)
            answer_1, answer_2 = qa_chain_1.run(q), qa_chain_2.run(q)
            summary_of_answers += "Question: " + q + "\n"
            summary_of_answers += f"{NAMES[0]} answer: " + answer_1 + f";\n {NAMES[1]} answer: " + answer_2 + "\n"
            st.write(NAMES[0], answer_1)
            st.write(NAMES[1], answer_2)
            st.write('-' * 10)

        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        len(encoder.encode(summary_of_answers))



        st.write(summary_of_answers)

        template = """
        I want you to act as an expert in insurance policies. I have asked two companies about their insurance policies and here are their answers:
        {summary_of_answers}
        I am looking for insurance for a full-remote consulting company with 100 employees. I want you to tell me which company is better and why.
        Give me a rating (x out of 10) for the following categories for each company separately with a short explanation (10 words max) for each category:
        1. Coverage of different health procedures
        2. Flexibility for remote workers abroad
        3. Price and compensation
        Your answer:
        """

        prompt = PromptTemplate(
            input_variables=["summary_of_answers"],
            template=template,
        )


        llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=OPENAI_API_KEY, request_timeout=60)
        chain = LLMChain(llm=llm, prompt=prompt)


        answer = chain.run(summary_of_answers)

        st.write(answer)








