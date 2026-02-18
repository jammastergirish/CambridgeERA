"""
Streamlit app for HuggingFace model inference.
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@st.cache_resource
def load_model(model_id: str):
    """Load model and tokenizer, auto-detecting device and dtype."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str, max_tokens: int = 200) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


st.title("HuggingFace Model Inference")

model_id = st.text_input(
    "Model ID (e.g., girishgupta/EleutherAI_deep-ignorance-unfiltered__cb_lat__ep2_lr1e-05_bs4_a100.0_sc20.0_le0.2_ls5_ly5-6-7)",
    placeholder="Enter HuggingFace model ID"
)

if model_id:
    with st.spinner(f"Loading {model_id}..."):
        try:
            model, tokenizer, device = load_model(model_id)
            st.success(f"Model loaded on {device}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    prompt = st.text_area("Enter your prompt:", height=100)

    col1, col2 = st.columns([1, 5])
    with col1:
        max_tokens = st.number_input("Max tokens", min_value=10, max_value=1000, value=200)

    if st.button("Generate", type="primary"):
        if prompt:
            with st.spinner("Generating..."):
                response = generate(model, tokenizer, prompt, device, max_tokens)
                st.markdown("### Response:")
                st.write(response)
        else:
            st.warning("Please enter a prompt")