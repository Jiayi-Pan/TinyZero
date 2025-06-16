#!/usr/bin/env python3
"""
Interactive Terminal Demo for Workfront AI Assistant
Perfect for cluster-only presentations - no web interface needed!
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
import os
import glob
from pathlib import Path

class WorkfrontTerminalDemo:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        
    def find_checkpoints(self):
        """Find available model checkpoints"""
        patterns = [
            "./checkpoints/*/actor/global_step_*",
            "./checkpoints_optimized/*/actor/global_step_*", 
            "./sft_*_model"
        ]
        
        checkpoints = []
        for pattern in patterns:
            checkpoints.extend(glob.glob(pattern))
        
        # Add base model
        checkpoints.append("Qwen/Qwen2.5-1.5B-Instruct")
        return sorted(checkpoints)
    
    def load_model(self, model_path):
        """Load model and tokenizer"""
        print(f"🔄 Loading model from {model_path}...")
        print("   This may take 30-60 seconds...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=False
            )
            self.model_path = model_path
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_prompt(self, question):
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
    
    def generate_response(self, prompt):
        """Generate response from model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        end_time = time.time()
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = full_response[len(prompt):].strip()
        
        return generated_part, end_time - start_time
    
    def extract_json(self, response):
        """Extract JSON from response"""
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
    
    def print_header(self):
        """Print demo header"""
        print("\n" + "="*80)
        print("🤖 WORKFRONT AI ASSISTANT - LIVE DEMO")
        print("="*80)
        print("🎯 Converting Natural Language → Workfront API Calls")
        print(f"🤖 Model: {Path(self.model_path).name}")
        print("="*80)
    
    def demo_query(self, question, show_thinking=False):
        """Demo a single query with nice formatting"""
        print(f"\n📝 QUESTION: {question}")
        print("-" * 60)
        
        # Generate response
        prompt = self.create_prompt(question)
        response, response_time = self.generate_response(prompt)
        
        print(f"⚡ Generated in {response_time:.2f} seconds")
        
        if show_thinking and "<thinking>" in response:
            thinking_start = response.find("<thinking>")
            thinking_end = response.find("</thinking>")
            if thinking_start != -1 and thinking_end != -1:
                thinking = response[thinking_start+10:thinking_end].strip()
                print(f"\n🧠 AI THINKING:")
                print(f"   {thinking}")
        
        # Extract and display JSON
        json_result = self.extract_json(response)
        
        if json_result:
            print(f"\n✅ GENERATED API CALL:")
            print(json.dumps(json_result, indent=2))
            
            # Validate structure
            required_keys = ["objCode", "fields", "filters"]
            missing_keys = [key for key in required_keys if key not in json_result]
            
            if not missing_keys:
                print("✅ Perfect JSON structure!")
            else:
                print(f"⚠️  Missing: {missing_keys}")
        else:
            print(f"\n❌ Could not extract valid JSON")
            print(f"Raw response: {response[:200]}...")
        
        print("="*60)
    
    def run_demo(self):
        """Run the interactive demo"""
        # Find and select model
        checkpoints = self.find_checkpoints()
        
        if not checkpoints:
            print("❌ No model checkpoints found!")
            return
        
        print("🔍 Available Models:")
        for i, checkpoint in enumerate(checkpoints):
            print(f"   {i+1}. {Path(checkpoint).name}")
        
        while True:
            try:
                choice = input(f"\nSelect model (1-{len(checkpoints)}): ")
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(checkpoints):
                    selected_model = checkpoints[model_idx]
                    break
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Please enter a number!")
        
        # Load model
        if not self.load_model(selected_model):
            return
        
        # Print header
        self.print_header()
        
        # Example queries for demo
        example_queries = [
            "Find all high priority tasks due this week",
            "Show me projects that are currently on hold",
            "Get users with email addresses containing '@company.com'",
            "What are the active tasks assigned to John Smith?",
            "Show me all projects with budget over $50000",
            "Find all users who are project managers",
            "Get tasks that are overdue and incomplete",
            "Show me all projects starting next month"
        ]
        
        print("\n🎪 DEMO MODE - Choose an option:")
        print("1. Run example queries")
        print("2. Interactive mode (type your own)")
        print("3. Presentation mode (auto-run examples)")
        
        mode = input("\nSelect mode (1-3): ")
        
        if mode == "1":
            # Show examples menu
            print("\n📋 Example Queries:")
            for i, query in enumerate(example_queries):
                print(f"   {i+1}. {query}")
            
            while True:
                try:
                    choice = input(f"\nSelect query (1-{len(example_queries)}) or 'q' to quit: ")
                    if choice.lower() == 'q':
                        break
                    query_idx = int(choice) - 1
                    if 0 <= query_idx < len(example_queries):
                        self.demo_query(example_queries[query_idx])
                    else:
                        print("Invalid choice!")
                except ValueError:
                    print("Please enter a number or 'q'!")
        
        elif mode == "2":
            # Interactive mode
            print("\n💬 Interactive Mode - Type your questions!")
            print("Type 'quit' to exit")
            
            while True:
                question = input("\n🤔 Your question: ")
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if question.strip():
                    self.demo_query(question, show_thinking=True)
        
        elif mode == "3":
            # Presentation mode
            print("\n🎬 PRESENTATION MODE")
            print("Running example queries automatically...")
            input("Press Enter to start...")
            
            for i, query in enumerate(example_queries[:5]):  # First 5 examples
                print(f"\n🎯 Example {i+1}/{5}")
                input("Press Enter for next query...")
                self.demo_query(query)
        
        print("\n🎉 Demo completed! Thank you!")

if __name__ == "__main__":
    demo = WorkfrontTerminalDemo()
    demo.run_demo() 