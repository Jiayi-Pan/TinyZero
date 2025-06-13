import json
import re
import random
import os
import datetime


def extract_answer(solution_str):
    """Extract the API request from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split("\n")[-1]

    final_json_pattern = r"<final_json>(.*?)</final_json>"
    match = re.finditer(final_json_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def extract_json(answer):
    """Extract ```json{}``` from the answer string."""
    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, answer)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


def is_valid_json(api_request):
    """Validate that the JSON string is valid."""
    try:
        json.loads(api_request)
        return True
    except:
        return False






def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    """Clean scoring function with proper logging for Workfront API task.

    Args:
        solution_str: the solution text from the model
        ground_truth: dictionary containing expected API call (or JSON string)
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    # Handle case where ground_truth is a JSON string
    if isinstance(ground_truth, str):
        try:
            expected_response = json.loads(ground_truth)
        except json.JSONDecodeError:
            print(f"❌ ERROR: Invalid ground_truth JSON: {ground_truth}")
            return 0
    else:
        expected_response = ground_truth
    
    # Setup logging to file
    log_dir = os.path.expanduser("/home/workfrontadmin/TinyZeroRL/workfront_training_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"reward_logs_{datetime.datetime.now().strftime('%Y%m%d')}.txt")
    
    def log_both(message):
        """Log to both console and file"""
        print(message)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            if message.startswith("\n="):
                f.write(f"\n[{timestamp}] {message}\n")
            else:
                f.write(f"[{timestamp}] {message}\n")
    
    # Extract model's answer
    answer = extract_answer(solution_str)
    
    log_both("\n" + "="*80)
    log_both("🤖 MODEL EVALUATION")
    log_both("="*80)
    
    # Show the question (extract from solution_str)
    if "User:" in solution_str:
        question = solution_str.split("User:")[-1].split("Assistant:")[0].strip()
        log_both(f"📝 QUESTION: {question}")
    
    log_both(f"\n🎯 EXPECTED RESPONSE:")
    log_both(json.dumps(expected_response, indent=2))
    
    if answer is None:
        log_both(f"\n❌ MODEL RESPONSE: No <answer> tags found")
        log_both(f"🔍 RAW OUTPUT: {solution_str[-200:]}")  # Last 200 chars
        log_both(f"⭐ REWARD SCORE: 0.0")
        log_both("="*80)
        return 0

    log_both(f"\n🤖 MODEL RESPONSE (Raw):")
    log_both(answer)
    
    # Try to extract JSON
    api_request = extract_json(answer)
    if not api_request:
        log_both(f"\n❌ EXTRACTED JSON: Invalid or missing JSON")
        log_both(f"⭐ REWARD SCORE: {format_score}")
        log_both("="*80)
        return format_score

    # log_both(f"\n✅ EXTRACTED JSON:")
    # log_both(json.dumps(api_request, indent=2))
    
    # Scoring logic
    score_breakdown = []
    final_score = 0.0
    
    # Check required fields presence
    required_fields = ["objCode", "fields", "filters"]  # Fixed field name
    missing_fields = []
    for field in required_fields:
        if field not in api_request:
            missing_fields.append(field)
    
    if missing_fields:
        final_score = format_score
        score_breakdown.append(f"❌ Missing fields: {missing_fields}")
    else:
        score_breakdown.append("✅ All required fields present")
        
        # Check objCode
        if api_request["objCode"] != expected_response["objCode"]:
            final_score = 0.4
            score_breakdown.append(f"❌ Wrong objCode: got '{api_request['objCode']}', expected '{expected_response['objCode']}'")
        else:
            score_breakdown.append("✅ Correct objCode")
            
            # Check filters (simplified comparison)
            model_filters = api_request.get("filters", {})
            expected_filters = expected_response.get("filters", {})
            
            if not model_filters and expected_filters:
                final_score = 0.5
                score_breakdown.append("❌ Missing filters")
            elif model_filters != expected_filters:
                final_score = 0.6
                score_breakdown.append("❌ Incorrect filters")
            else:
                score_breakdown.append("✅ Correct filters")
                
                # Check fields
                model_fields = set(api_request.get("fields", []))
                expected_fields = set(expected_response.get("fields", []))
                
                # Allow some flexibility for ID and name fields
                core_expected = expected_fields - {"ID", "name"}
                core_model = model_fields - {"ID", "name"}
                
                if core_expected.issubset(core_model):
                    final_score = 1.0
                    score_breakdown.append("✅ All important fields present")
                else:
                    final_score = 0.7
                    missing = core_expected - core_model
                    score_breakdown.append(f"❌ Missing important fields: {missing}")
    
    log_both(f"\n📊 SCORING BREAKDOWN:")
    for item in score_breakdown:
        log_both(f"   {item}")
    
    log_both(f"\n⭐ FINAL REWARD SCORE: {final_score}")
    log_both("="*80)
    
    return final_score 