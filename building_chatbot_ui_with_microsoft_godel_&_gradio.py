# -*- coding: utf-8 -*-




GODEL - (Grounded Open
Dialogue Language Model https://www.microsoft.com/en-us/research/uploads/prod/2022/05/2206.11309.pdf
"""

! pip install transformers gradio -q

!pip install huggingface_hub
from huggingface_hub import notebook_login

# Log in to Hugging Face
notebook_login()

"""# Step 1 — Setting up the Chatbot Model - Microsoft phi-3.5"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

"""# Step 2 — Defining a `predict` function with `state` and model prediction"""

def predict(input, history=[]):

    instruction = 'Instruction: given a dialog context, you need to response empathically'

    knowledge = '  '

    s = list(sum(history, ()))

    s.append(input)

    #print(s)

    dialog = ' EOS ' .join(s)

    #print(dialog)

    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"

    top_p = 0.9
    min_length = 8
    max_length = 64


    # tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(f"{query}", return_tensors='pt')


    output = model.generate(new_user_input_ids, min_length=int(
        min_length), max_length=int(max_length), top_p=top_p, do_sample=True).tolist()


    response = tokenizer.decode(output[0], skip_special_tokens=True)


    history.append((input, response))

    return history, history

"""# Step 3 — Creating a Gradio Chatbot UI"""

import gradio as gr


gr.Interface(fn=predict,
             inputs=["text",'state'],
             outputs=["chatbot",'state']).launch(debug = True, share = True)

