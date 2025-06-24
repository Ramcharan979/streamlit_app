
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model.eval(), tokenizer

model, tokenizer = load_model()

st.title("🧠 AI Text Generator with GPT-2")
st.markdown("Generate intelligent text completions using your fine-tuned Transformer model 🚀")

prompt = st.text_input("Enter a prompt:", "Once upon a time,")
max_length = st.slider("Max output length", 20, 200, 100)
temperature = st.slider("Creativity (temperature)", 0.5, 1.5, 0.7)

if st.button("🔮 Generate"):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        top_p=0.95,
        do_sample=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success("✅ Here's your generated text:")
    st.write(result)
