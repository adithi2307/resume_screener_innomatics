import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from prompts.prompts import extraction_prompt, evaluation_prompt
from chains.chains import create_extraction_chain, create_evaluation_chain

# Load secrets
load_dotenv()

def main():
    # 1. Setup the Free Endpoint
    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational", # Changed task to match server requirements
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.1,
    )

    # 2. Wrap it in ChatHuggingFace (This fixes the 'featherless-ai' error)
    llm = ChatHuggingFace(llm=llm_endpoint)

    # 3. Define the Job Description
    job_description = """
    We are looking for a Data Scientist with 2+ years of experience.
    Required skills: Python, SQL, LangChain, and Machine Learning.
    Must have experience building LLM pipelines.
    """

    # 4. Define 3 Candidates
    candidates = [
        {"name": "Strong Candidate", "text": "Adithi Onkar. 3 years exp. Skills: Python, SQL, LangChain, PyTorch. Built LLM apps."},
        {"name": "Average Candidate", "text": "John Doe. 1 year exp. Skills: Python, Java, Basic Machine Learning and SQL."},
        {"name": "Weak Candidate", "text": "Jane Smith. 5 years in Graphic Design. Skills: Photoshop, Illustrator, Canva."}
    ]

    # 5. Create the chains
    extract_chain = create_extraction_chain(llm, extraction_prompt)
    eval_chain = create_evaluation_chain(llm, evaluation_prompt)

    # 6. Run the pipeline
    for person in candidates:
        print(f"\n--- Processing: {person['name']} ---")
        try:
            # Step 1: Extract
            info = extract_chain.invoke({"resume_text": person['text']})
            
            # Step 2: Evaluate
            final_report = eval_chain.invoke({
                "extracted_info": info,
                "job_description": job_description
            })
            
            # Since it's a ChatModel, we print the content of the message
            print(final_report)
        except Exception as e:
            print(f"Error processing {person['name']}: {e}")

if __name__ == "__main__":
    main()