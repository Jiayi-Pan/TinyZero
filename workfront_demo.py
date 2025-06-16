#!/usr/bin/env python3

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import glob
from pathlib import Path
import time

# Page config
st.set_page_config(
    page_title="Workfront AI Assistant",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_model(model_path):
    """Load model and tokenizer (cached for performance)"""
    try:
        st.info(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False
        )
        st.success("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def find_available_checkpoints():
    """Find all available model checkpoints"""
    checkpoint_patterns = [
        "./checkpoints/*/actor/global_step_*",
        "./sft_*_model",
        "./checkpoints/TinyZero/*/actor/global_step_*"
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(glob.glob(pattern))
    
    # Add base model option
    checkpoints.append("Qwen/Qwen2.5-1.5B-Instruct")
    
    return sorted(checkpoints)

def create_prompt(question):
    """Create prompt for the model"""
    return f"""You are a Workfront API assistant. Convert natural language queries into structured JSON API calls.

TASK: Convert this query to Workfront API JSON:
"{question}"

Required JSON format:
```json
{{
  "objCode": "TASK|PROJ|USER",
  "fields": ["relevant", "fields"],
  "filters": {{"condition": "value"}}
}}
```

Answer:"""

def generate_response(model, tokenizer, prompt, max_tokens=512):
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_part = full_response[len(prompt):].strip()
    return generated_part

def extract_json_from_response(response):
    """Extract JSON from model response"""
    try:
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                json_str = response[start:end].strip()
                return json.loads(json_str)
        
        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
            return json.loads(json_str)
            
        return None
    except json.JSONDecodeError:
        return None

# Main app
def main():
    st.title("🤖 Workfront AI Assistant")
    st.markdown("Convert natural language queries into Workfront API calls using our trained RL model!")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    # Find available checkpoints
    checkpoints = find_available_checkpoints()
    
    if not checkpoints:
        st.error("No model checkpoints found!")
        st.info("Make sure your training has saved some checkpoints with trainer.save_freq > 0")
        return
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model Checkpoint:",
        checkpoints,
        help="Choose a trained model checkpoint or the base model"
    )
    
    # Load model
    if 'current_model_path' not in st.session_state or st.session_state.current_model_path != selected_model:
        st.session_state.current_model_path = selected_model
        st.session_state.model = None
        st.session_state.tokenizer = None
    
    if st.session_state.model is None:
        model, tokenizer = load_model(selected_model)
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
    
    if st.session_state.model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Natural Language Query")
        
        # Predefined examples
        example_queries = [
            "Find all high priority tasks due this week",
            "Show me projects that are currently on hold", 
            "Get users with email addresses containing '@company.com'",
            "What are the active tasks assigned to John Smith?",
            "Show me all projects with budget over $50000",
            "Find all users who are project managers",
            "Get tasks that are overdue and incomplete"
        ]
        
        selected_example = st.selectbox("Choose an example:", ["Custom query..."] + example_queries)
        
        if selected_example != "Custom query...":
            query = st.text_area("Your Query:", value=selected_example, height=100)
        else:
            query = st.text_area("Your Query:", placeholder="Enter your Workfront query here...", height=100)
        
        if st.button("🚀 Generate API Call", type="primary"):
            if query.strip():
                with st.spinner("AI is thinking..."):
                    # Generate response
                    prompt = create_prompt(query)
                    start_time = time.time()
                    response = generate_response(st.session_state.model, st.session_state.tokenizer, prompt)
                    end_time = time.time()
                    
                    # Store results in session state
                    st.session_state.last_query = query
                    st.session_state.last_response = response
                    st.session_state.last_time = end_time - start_time
            else:
                st.warning("Please enter a query!")
    
    with col2:
        st.header("🎯 Generated API Call")
        
        if 'last_response' in st.session_state:
            # Show timing
            st.info(f"⚡ Generated in {st.session_state.last_time:.2f} seconds")
            
            # Extract and display JSON
            json_result = extract_json_from_response(st.session_state.last_response)
            
            if json_result:
                st.success("✅ Valid JSON Generated!")
                st.json(json_result)
                
                # Validation
                required_keys = ["objCode", "fields", "filters"]
                missing_keys = [key for key in required_keys if key not in json_result]
                
                if not missing_keys:
                    st.success("✅ All required fields present!")
                else:
                    st.warning(f"⚠️ Missing fields: {missing_keys}")
                
            else:
                st.error("❌ Could not extract valid JSON")
                st.text("Raw Response:")
                st.code(st.session_state.last_response)
        else:
            st.info("👆 Enter a query above to see the generated API call")
    
    # Show model info
    st.sidebar.markdown("---")
    st.sidebar.header("Model Info")
    st.sidebar.info(f"**Current Model:** {Path(selected_model).name}")
    
    if 'last_time' in st.session_state:
        st.sidebar.metric("Response Time", f"{st.session_state.last_time:.2f}s")
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Select a Model**: Choose from available checkpoints in the sidebar
        2. **Enter Query**: Type a natural language Workfront query
        3. **Generate**: Click the button to get the API call
        4. **Review**: Check the generated JSON for accuracy
        
        **Example Queries:**
        - "Find tasks assigned to [name]"
        - "Show projects with status [status]"
        - "Get users in [group/team]"
        - "Find overdue/high priority items"
        """)

if __name__ == "__main__":
    main() 