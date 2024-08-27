import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import gradio as gr
import os
from functools import lru_cache


from threading import Thread
import subprocess
import logging
subprocess.run('pip install -U flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

models_available = [
    "MohamedRashad/Arabic-Orpo-Llama-3-8B-Instruct",
    "silma-ai/SILMA-9B-Instruct-v1.0",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "MaziyarPanahi/calme-2.2-qwen2-72b",
    "davidkim205/Rhea-72b-v0.5",
    "dnhkng/RYS-XLarge",
    "arcee-ai/Arcee-Nova",
    "paloalma/TW3-JRGL-v2",
    "freewheelin/free-evo-qwen72b-v0.8-re",
    "dfurman/Qwen2-72B-Orpo-v0.1",
    "MaziyarPanahi/calme-2.1-qwen2-72b",
    "UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3",
    ""
    "inceptionai/jais-adapted-7b-chat",
    "inceptionai/jais-family-6p7b-chat",
    "inceptionai/jais-family-2p7b-chat",
    "inceptionai/jais-family-1p3b-chat",
    "inceptionai/jais-family-590m-chat",
]

tokenizer_a, model_a = None, None
tokenizer_b, model_b = None, None
torch_dtype = torch.bfloat16
attn_implementation = "flash_attention_2"

def load_model_a(model_id):
    global tokenizer_a, model_a
    tokenizer_a = AutoTokenizer.from_pretrained(model_id)
    print(f"model A: {tokenizer_a.eos_token}")
    try:
        model_a = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        ).eval()
    except Exception as e:
        print(f"Using default attention implementation in {model_id}")
        print(f"Error: {e}")
        model_a = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    model_a.tie_weights()
    return gr.update(label=model_id)
    
def load_model_b(model_id):
    global tokenizer_b, model_b
    tokenizer_b = AutoTokenizer.from_pretrained(model_id)
    print(f"model B: {tokenizer_b.eos_token}")
    try:
        model_b = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        ).eval()
    except Exception as e:
        print(f"Error: {e}")
        print(f"Using default attention implementation in {model_id}")
        model_b = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    model_b.tie_weights()
    return gr.update(label=model_id)

@spaces.GPU()
def generate_both(system_prompt, input_text, chatbot_a, chatbot_b, max_new_tokens=2048, temperature=0.2, top_p=0.9, repetition_penalty=1.1):

    text_streamer_a = TextIteratorStreamer(tokenizer_a, skip_prompt=True)
    text_streamer_b = TextIteratorStreamer(tokenizer_b, skip_prompt=True)

    system_prompt_list = [{"role": "system", "content": system_prompt}] if system_prompt else []
    input_text_list = [{"role": "user", "content": input_text}]

    chat_history_a = []
    for user, assistant in chatbot_a:
        chat_history_a.append({"role": "user", "content": user})
        chat_history_a.append({"role": "assistant", "content": assistant})

    chat_history_b = []
    for user, assistant in chatbot_b:
        chat_history_b.append({"role": "user", "content": user})
        chat_history_b.append({"role": "assistant", "content": assistant})
    
    base_messages = system_prompt_list + chat_history_a + input_text_list
    new_messages = system_prompt_list + chat_history_b + input_text_list

    input_ids_a = tokenizer_a.apply_chat_template(
        base_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model_a.device)

    input_ids_b = tokenizer_b.apply_chat_template(
        new_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model_b.device)

    generation_kwargs_a = dict(
        input_ids=input_ids_a,
        streamer=text_streamer_a,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer_a.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    generation_kwargs_b = dict(
        input_ids=input_ids_b,
        streamer=text_streamer_b,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer_b.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    thread_a = Thread(target=model_a.generate, kwargs=generation_kwargs_a)
    thread_b = Thread(target=model_b.generate, kwargs=generation_kwargs_b)

    thread_a.start()
    thread_b.start()

    chatbot_a.append([input_text, ""])
    chatbot_b.append([input_text, ""])

    finished_a = False
    finished_b = False

    while not (finished_a and finished_b):
        if not finished_a:
            try:
                text_a = next(text_streamer_a)
                if tokenizer_a.eos_token in text_a:
                    eot_location = text_a.find(tokenizer_a.eos_token)
                    text_a = text_a[:eot_location]
                    finished_a = True
                chatbot_a[-1][-1] += text_a
                yield chatbot_a, chatbot_b
            except StopIteration:
                finished_a = True

        if not finished_b:
            try:
                text_b = next(text_streamer_b)
                if tokenizer_b.eos_token in text_b:
                    eot_location = text_b.find(tokenizer_b.eos_token)
                    text_b = text_b[:eot_location]
                    finished_b = True
                chatbot_b[-1][-1] += text_b
                yield chatbot_a, chatbot_b
            except StopIteration:
                finished_b = True

    return chatbot_a, chatbot_b

def clear():
    return [], []

arena_notes = """## Important Notes:
- Sometimes an error may occur when generating the response, in this case, please try again.
"""

with gr.Blocks() as demo:
    with gr.Column():
        gr.HTML("<center><h1>Arabic Chatbot Comparison</h1></center>")
        gr.Markdown(arena_notes)
        system_prompt = gr.Textbox(lines=1, label="System Prompt", value="أنت متحدث لبق باللغة العربية!", rtl=True, text_align="right", show_copy_button=True)
        with gr.Row(variant="panel"):
            with gr.Column():
                model_dropdown_a = gr.Dropdown(label="Model A", choices=models_available, value=None)
                chatbot_a = gr.Chatbot(label="Model A", rtl=True, likeable=True, show_copy_button=True, height=500)
            with gr.Column():
                model_dropdown_b = gr.Dropdown(label="Model B", choices=models_available, value=None)
                chatbot_b = gr.Chatbot(label="Model B", rtl=True, likeable=True, show_copy_button=True, height=500)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                submit_btn = gr.Button(value="Generate", variant="primary")
                clear_btn = gr.Button(value="Clear", variant="secondary")
            input_text = gr.Textbox(lines=1, label="", value="مرحبا", rtl=True, text_align="right", scale=3, show_copy_button=True)
        with gr.Accordion(label="Generation Configurations", open=False):
            max_new_tokens = gr.Slider(minimum=128, maximum=4096, value=2048, label="Max New Tokens", step=128)
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, label="Temperature", step=0.01)
            top_p = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="Top-p", step=0.01)
            repetition_penalty = gr.Slider(minimum=0.1, maximum=2.0, value=1.1, label="Repetition Penalty", step=0.1)

    model_dropdown_a.change(load_model_a, inputs=[model_dropdown_a], outputs=[chatbot_a])
    model_dropdown_b.change(load_model_b, inputs=[model_dropdown_b], outputs=[chatbot_b])

    input_text.submit(generate_both, inputs=[system_prompt, input_text, chatbot_a, chatbot_b, max_new_tokens, temperature, top_p, repetition_penalty], outputs=[chatbot_a, chatbot_b])
    submit_btn.click(generate_both, inputs=[system_prompt, input_text, chatbot_a, chatbot_b, max_new_tokens, temperature, top_p, repetition_penalty], outputs=[chatbot_a, chatbot_b])
    clear_btn.click(clear, outputs=[chatbot_a, chatbot_b])

if __name__ == "__main__":
    demo.queue().launch()