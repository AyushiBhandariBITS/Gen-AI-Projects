import torch
import os
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.utils import enforce_stop_tokens
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

token= {
    "url"    : ""
    }
params = {
        GenParams.MAX_NEW_TOKENS: 800, 
        GenParams.TEMPERATURE: 0.1,   
    }

LLAMA2_model = Model(
        model_id= 'meta-llama/llama-3-8b-chat', 
        credentials=token,
        params=params,
        project_id="skills-network",  
        )

llm = WatsonxLLM(LLAMA2_model)  


temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""

pt = PromptTemplate(
    input_variables=["context"],
    template= temp)

prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)


def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
    result = prompt_to_LLAMA2.run(transcript_txt)

    return result

