from langchain_core.prompts import ChatPromptTemplate

# Step 1: Extracting skills
extraction_template = """
You are an expert HR Assistant. Extract the following from the resume text:
1. Technical Skills
2. Years of Experience
3. Main Tools mentioned

Resume: {resume_text}

Output the result clearly. Do NOT assume skills not written in the text.
"""
extraction_prompt = ChatPromptTemplate.from_template(extraction_template)

# Step 2: Scoring and Matching
evaluation_template = """
You are a Senior Data Science Recruiter. 
Compare the 'Extracted Skills' with the 'Job Description'.

Extracted Skills: {extracted_info}
Job Description: {job_description}

Provide:
1. Fit Score: (0 to 100)
2. Reasoning: Explain exactly why you gave that score. 
3. Missing Skills: List what is missing from the JD.
"""
evaluation_prompt = ChatPromptTemplate.from_template(evaluation_template)