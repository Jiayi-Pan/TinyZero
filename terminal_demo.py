import re
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
        trained_checkpoints = []

        # Look for your specific checkpoint structure
        checkpoint_patterns = [
            "./checkpoints/*/actor/global_step_*",
            "./checkpoints_optimized/*/actor/global_step_*",
            "/home/workfrontadmin/checkpoints/TinyZero/*/actor/global_step_*",
            "./sft_*_model",
        ]

        for pattern in checkpoint_patterns:
            found = glob.glob(pattern)
            trained_checkpoints.extend(found)

        # Remove duplicates and sort by step number
        trained_checkpoints = list(set(trained_checkpoints))
        trained_checkpoints = sorted(
            trained_checkpoints,
            key=lambda x: (
                int(x.split("global_step_")[-1]) if "global_step_" in x else 0
            ),
        )

        print("🔍 Found trained checkpoints:", trained_checkpoints)

        # Add base model at the end
        all_checkpoints = trained_checkpoints + ["Qwen/Qwen2.5-1.5B-Instruct"]
        return all_checkpoints

    def load_model(self, model_path):
        """Load model and tokenizer"""
        print(f"🔄 Loading model from {model_path}...")
        print("   This may take 30-60 seconds...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"✅ Tokenizer loaded - vocab size: {len(self.tokenizer)}")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=False,
            )
            self.model_path = model_path

            # Debug model info
            print(f"✅ Model loaded - parameters: {self.model.num_parameters():,}")
            print(f"🔍 Model config: {self.model.config.model_type}")
            print(f"🔍 Model vocab size: {self.model.config.vocab_size}")
            print(f"🔍 Tokenizer vocab size: {len(self.tokenizer)}")

            # Check if vocab sizes match
            if self.model.config.vocab_size != len(self.tokenizer):
                print("⚠️  WARNING: Model and tokenizer vocab sizes don't match!")

            # Test a simple generation
            print("🧪 Testing simple generation...")
            test_input = "Hello"
            test_tokens = self.tokenizer(test_input, return_tensors="pt")
            print(f"Model device: {self.model.device}")
            test_tokens = {k: v.to(self.model.device) for k, v in test_tokens.items()}

            with torch.no_grad():
                test_output = self.model.generate(
                    **test_tokens,
                    max_new_tokens=200,
                    # do_sample=False,
                    # pad_token_id=self.tokenizer.eos_token_id,
                )

            test_response = self.tokenizer.decode(
                test_output[0], skip_special_tokens=True
            )
            print(f"🧪 Test generation: '{test_response}'")

            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback

            traceback.print_exc()
            return False

    def create_prompt(self, question):
        """Create prompt for the model using the EXACT same format as training data"""
        # Load context files (same as in fetch_data.py)
        try:
            with open("verl/utils/dataset/context.txt", "r") as f:
                workfront_context = f.read().strip()
            print(f"🔍 DEBUG: Loaded context.txt ({len(workfront_context)} chars)")
        except Exception as e:
            workfront_context = "Workfront API Context - Context file not found"
            print(f"⚠️ DEBUG: Could not load context.txt: {e}")

        # Simple obj_code detection for context
        obj_code = "PROJ"  # Default to PROJ for most queries
        if any(word in question.lower() for word in ["task", "tasks"]):
            obj_code = "TASK"
        elif any(
            word in question.lower() for word in ["user", "person", "people", "role"]
        ):
            obj_code = "USER"

        try:
            with open(f"verl/utils/dataset/{obj_code.lower()}_context.txt", "r") as f:
                obj_code_context = f.read().strip()
            print(
                f"🔍 DEBUG: Loaded {obj_code.lower()}_context.txt ({len(obj_code_context)} chars)"
            )
        except Exception as e:
            obj_code_context = f"{obj_code} context not found"
            print(f"⚠️ DEBUG: Could not load {obj_code.lower()}_context.txt: {e}")

        # Use the EXACT format from fetch_data.py make_prefix function
        prompt = f"""<|im_start|>system
You are a helpful AI assistant designed to convert natural language queries into structured JSON commands for querying the Workfront project management system. You use Workfront's custom object names and metadata to do the same using the context given below.

Your role is to interpret a user's natural language request, determine the correct object (objCode like TASK, PROJ, or USER), extract relevant fields (the attributes to display), and construct appropriate filters (conditions the data must satisfy). 

You will take the user's natural language prompt and give a structured JSON response. ALWAYS include just the final JSON with the correct json structure in <final_json> tags. The tags should always be called <final_json> and always inside tags use ```json``` to indicate the json structure. 
USE STRUCTURE EXACTLY LIKE BELOW:

<final_json>
```json
{{
  "objCode": "TASK | PROJ | USER", // Choose based on what the user is asking about
  "fields": [],        // Include ALL relevant fields mentioned in the query      
  "filters": {{}} // Include ALL conditions mentioned in the query
}}
```
</final_json>
The JSON must be wrapped in triple backticks to indicate code formatting.

Here are some examples:

Example 1:
User Prompt: What are all the tasks with high priority due next week?

Assistant:
<final_json>
```json
{{
  "objCode": "TASK",
  "fields": ["ID", "name", "priority", "plannedCompletionDate"],
  "filters": {{
        "priority": 3,
        "actualCompletionDate_Mod": "isnull",
        "plannedCompletionDate": "$$TODAYb+1w",
        "plannedCompletionDate_Mod": "between",
        "plannedCompletionDate_Range": "$$TODAYe+1w"
  }}
}}
```
</final_json>

Example 2:
User Prompt: Show me all projects that are currently on hold

Assistant:
<final_json>
```json
{{
  "objCode": "PROJ",
  "fields": ["ID", "name", "status", "plannedCompletionDate"],
  "filters": {{
        "status": "OHD"
  }}
}}
```
</final_json>

Example 3:
User Prompt: Find users with email addresses containing '@company.com'

Assistant:
<final_json>
```json
{{
  "objCode": "USER",
  "fields": ["ID", "name", "emailAddr", "username"],
  "filters": {{
        "emailAddr_Mod": "cicontains",
        "emailAddr": "@company.com"
  }}
}}
```
</final_json>

{workfront_context}
{obj_code_context}

<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
I'll help you with defining the correct JSON object with the correct objCode, fields, and filters.

<thinking>
I need to understand the user's request and determine:
1. Which objCode they are asking about
2. What specific fields they need to see
3. What conditions (filters) they want to apply
"""

        print(f"🔍 DEBUG: Using training prompt format ({len(prompt)} chars)")
        return prompt

    def extract_text_after_thinking(self, solution_str):
        # Extract all text after the <thinking> tag
        thinking_start = solution_str.find("<thinking>")
        position_to_slice = thinking_start
        return solution_str[position_to_slice:].strip()

    def generate_response(self, prompt):
        """Generate response from model"""
        print(f"🔍 DEBUG: Prompt length: {len(prompt)} chars")
        print(f"🔍 DEBUG: Prompt preview: {prompt[:100]}...")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        print(f"🔍 DEBUG: Input tokens shape: {inputs['input_ids'].shape}")
        print(f"🔍 DEBUG: First 10 token IDs: {inputs['input_ids'][0][:10].tolist()}")

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                # do_sample=True,
                # temperature=0.1,
                # top_p=0.9,
                # pad_token_id=self.tokenizer.eos_token_id,
                # eos_token_id=self.tokenizer.eos_token_id,
            )
        end_time = time.time()

        print(f"🔍 DEBUG: Output tokens shape: {outputs.shape}")
        print(
            f"🔍 DEBUG: Generated token IDs: {outputs[0][inputs['input_ids'].shape[1]:inputs['input_ids'].shape[1]+10].tolist()}"
        )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # generated_part = full_response[len(prompt) :].strip()
        generated_part = self.extract_text_after_thinking(full_response.strip())

        print(f"🔍 DEBUG: Full response length: {len(full_response)}")
        print(f"🔍 DEBUG: Generated part length: {len(generated_part)}")

        return generated_part, end_time - start_time

    def strip_json_comments(self, json_str):
        """Remove // and /* */ style comments from JSON string."""
        # Remove single-line comments (// comment)
        json_str = re.sub(r"//.*?(?=\n|$)", "", json_str)

        # Remove multi-line comments (/* comment */)
        json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

        return json_str

    def extract_json_v2(self, answer):
        """Extract ```json{}``` from the answer string."""
        # Use DOTALL flag to make . match newlines as well
        json_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_pattern, answer, re.DOTALL)
        if match:
            json_str = match.group(1).strip()

            # Try parsing as-is first
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If that fails, try stripping comments
                try:
                    cleaned_json = self.strip_json_comments(json_str)
                    return json.loads(cleaned_json)
                except json.JSONDecodeError:
                    return None
        return None

    def extract_json(self, response):
        """Extract JSON from response - looking for <final_json> tags as trained"""
        try:
            print(f"🔍 DEBUG: Response length: {len(response)} chars")
            # print(f"🔍 DEBUG: First 200 chars: {response[200:]
            print(f"🔍 DEBUG: Response: {response}")

            # First try to find <final_json> tags (as the model was trained)
            if "<final_json>" in response and "</final_json>" in response:
                start_tag = response.find("<final_json>")
                end_tag = response.find("</final_json>")
                if start_tag != -1 and end_tag != -1:
                    json_section = response[start_tag + 12 : end_tag].strip()
                    print(
                        f"🔍 DEBUG: Found final_json section: {json_section[:100]}..."
                    )

                    # Now extract the JSON from within the ```json``` blocks
                    if "```json" in json_section:
                        json_start = json_section.find("```json") + 7
                        json_end = json_section.find("```", json_start)
                        if json_end != -1:
                            json_str = json_section[json_start:json_end].strip()
                            print(f"🔍 DEBUG: Extracted JSON: {json_str[:100]}...")
                            return json.loads(json_str)

            # Fallback: Look for ```json blocks anywhere
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end != -1:
                    json_str = response[start:end].strip()
                    print(f"🔍 DEBUG: Fallback JSON extraction: {json_str[:100]}...")
                    return json.loads(json_str)

            # Last resort: Look for any JSON-like structure
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                print(f"🔍 DEBUG: Last resort JSON: {json_str[:100]}...")
                return json.loads(json_str)

            print("🔍 DEBUG: No JSON structure found")
            return None
        except json.JSONDecodeError as e:
            print(f"🔍 DEBUG: JSON decode error: {e}")
            return None

    def print_header(self):
        """Print demo header"""
        print("\n" + "=" * 80)
        print("🤖 WORKFRONT AI ASSISTANT - LIVE DEMO")
        print("=" * 80)
        print("🎯 Converting Natural Language → Workfront API Calls")
        print(f"🤖 Model: {Path(self.model_path).name}")
        print("=" * 80)

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
                thinking = response[thinking_start + 10 : thinking_end].strip()
                print(f"\n🧠 AI THINKING:")
                print(f"   {thinking}")

        # Extract and display JSON
        json_result = self.extract_json_v2(response)

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

        print("=" * 60)

    def run_demo(self):
        """Run the interactive demo"""
        # Find and select model
        checkpoints = self.find_checkpoints()

        if not checkpoints:
            print("❌ No model checkpoints found!")
            return

        print("🔍 Available Models:")
        for i, checkpoint in enumerate(checkpoints):
            if "global_step_" in checkpoint:
                step_num = checkpoint.split("global_step_")[-1]
                checkpoint_name = (
                    checkpoint.split("/")[-3]
                    if len(checkpoint.split("/")) > 3
                    else "checkpoint"
                )
                print(
                    f"   {i+1}. 🟢 TRAINED MODEL - Step {step_num} ({checkpoint_name})"
                )
            elif checkpoint == "Qwen/Qwen2.5-1.5B-Instruct":
                print(f"   {i+1}. 🔵 BASE MODEL - {checkpoint}")
            else:
                print(f"   {i+1}. {Path(checkpoint).name}")

        while True:
            try:
                choice = input(f"\nSelect model (1-{len(checkpoints)}): ")
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(checkpoints):
                    selected_model = checkpoints[model_idx]
                    print(f"🔍 DEBUG: Selected model: {selected_model}")
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
            "Show me all projects starting next month",
        ]

        print("\n🎪 DEMO MODE - Choose an option:")
        print("1. Run example queries")
        print("2. Interactive mode (type your own)")
        print("3. Presentation mode (auto-run examples)")
        print("4. Compare with base model (same query, both models)")

        mode = input("\nSelect mode (1-4): ")

        if mode == "1":
            # Show examples menu
            print("\n📋 Example Queries:")
            for i, query in enumerate(example_queries):
                print(f"   {i+1}. {query}")

            while True:
                try:
                    choice = input(
                        f"\nSelect query (1-{len(example_queries)}) or 'q' to quit: "
                    )
                    if choice.lower() == "q":
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
                if question.lower() in ["quit", "exit", "q"]:
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

        elif mode == "4":
            # Comparison mode
            print("\n🔄 COMPARISON MODE")
            print(
                "This will test the same query on both your trained model and the base model"
            )

            # Load base model for comparison
            base_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
            print(f"\n🔄 Loading base model: {base_model_path}")
            print("   This may take 30-60 seconds...")
            print(f"🔍 DEBUG: Your trained model path: {self.model_path}")
            print(f"🔍 DEBUG: Base model path: {base_model_path}")

            try:
                base_tokenizer = AutoTokenizer.from_pretrained(
                    base_model_path, cache_dir=None, force_download=False
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    cache_dir=None,
                    force_download=False,
                )
                print("✅ Base model loaded!")
                print(f"🔍 DEBUG: Base model config: {base_model.config.name_or_path}")
                print(
                    f"🔍 DEBUG: Trained model config: {self.model.config.name_or_path}"
                )

                # Check if models are actually different
                base_param_count = sum(p.numel() for p in base_model.parameters())
                trained_param_count = sum(p.numel() for p in self.model.parameters())
                print(f"🔍 DEBUG: Base model parameters: {base_param_count:,}")
                print(f"🔍 DEBUG: Trained model parameters: {trained_param_count:,}")

                # Check a few parameter values to see if they're different
                base_first_param = next(base_model.parameters()).flatten()[:5]
                trained_first_param = next(self.model.parameters()).flatten()[:5]
                print(f"🔍 DEBUG: Base model first 5 params: {base_first_param}")
                print(f"🔍 DEBUG: Trained model first 5 params: {trained_first_param}")

                params_different = not torch.allclose(
                    base_first_param, trained_first_param, atol=1e-6
                )
                print(f"🔍 DEBUG: Models have different parameters: {params_different}")
            except Exception as e:
                print(f"❌ Error loading base model: {e}")
                return

            # Get query from user
            question = input("\n🤔 Enter your question to compare both models: ")
            if not question.strip():
                print("No question provided!")
                return

            print(f"\n📝 COMPARING: {question}")
            print("=" * 80)

            # Test base model first
            print("🔵 BASE MODEL (Pretrained)")
            print("-" * 40)
            prompt = self.create_prompt(question)

            # Generate with base model
            inputs = base_tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}

            start_time = time.time()
            with torch.no_grad():
                outputs = base_model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    # do_sample=True,
                    # temperature=0.1,
                    # top_p=0.9,
                    # pad_token_id=base_tokenizer.eos_token_id,
                )
            end_time = time.time()

            base_response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)
            base_generated = base_response[len(prompt) :].strip()
            base_time = end_time - start_time
            base_json = self.extract_json_v2(base_generated)

            if base_json:
                print(f"✅ Generated in {base_time:.2f}s")
                print(json.dumps(base_json, indent=2))
            else:
                print(f"❌ Failed to generate valid JSON ({base_time:.2f}s)")
                print(f"Raw: {base_generated[:150]}...")

            print("\n" + "-" * 40)

            # Test trained model
            print("🟢 TRAINED MODEL (Your Model)")
            print("-" * 40)
            trained_response, trained_time = self.generate_response(prompt)
            trained_json = self.extract_json_v2(trained_response)

            if trained_json:
                print(f"✅ Generated in {trained_time:.2f}s")
                print(json.dumps(trained_json, indent=2))
            else:
                print(f"❌ Failed to generate valid JSON ({trained_time:.2f}s)")
                print(f"Raw: {trained_response[:150]}...")

            # Summary
            print("\n🏆 COMPARISON SUMMARY")
            print("-" * 40)
            base_success = base_json is not None
            trained_success = trained_json is not None

            print(
                f"Base Model:    {'✅ Success' if base_success else '❌ Failed'} ({base_time:.2f}s)"
            )
            print(
                f"Trained Model: {'✅ Success' if trained_success else '❌ Failed'} ({trained_time:.2f}s)"
            )

            if trained_success and not base_success:
                print(
                    "🎉 IMPROVEMENT: Your trained model succeeded where base model failed!"
                )
            elif trained_success and base_success:
                print("🎯 BOTH SUCCESSFUL: Compare the JSON quality above")
            elif not trained_success and base_success:
                print("⚠️  Base model performed better on this query")
            else:
                print("❌ Both models failed on this query")

            print("=" * 80)

        print("\n🎉 Demo completed! Thank you!")


if __name__ == "__main__":
    demo = WorkfrontTerminalDemo()
    demo.run_demo()
