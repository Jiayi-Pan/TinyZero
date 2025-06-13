import json
import re
import random


def extract_answer(solution_str):
    """Extract the API request from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        return final_answer
    else:
        return None


def extract_json(answer):
    """Extract ```json{}``` from the answer string."""
    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, answer, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    """Clean scoring function with proper logging for Workfront API task.

    Args:
        solution_str: the solution text from the model
        ground_truth: dictionary containing expected API call
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    expected_response = ground_truth
    
    # Extract model's answer
    answer = extract_answer(solution_str)
    
    print("\n" + "="*80)
    print("🤖 MODEL EVALUATION")
    print("="*80)
    
    # Show the question (extract from solution_str)
    if "User:" in solution_str:
        question = solution_str.split("User:")[-1].split("Assistant:")[0].strip()
        print(f"📝 QUESTION: {question}")
    
    print(f"\n🎯 EXPECTED RESPONSE:")
    print(json.dumps(expected_response, indent=2))
    
    if answer is None:
        print(f"\n❌ MODEL RESPONSE: No <answer> tags found")
        print(f"🔍 RAW OUTPUT: {solution_str[-200:]}")  # Last 200 chars
        print(f"⭐ REWARD SCORE: 0.0")
        print("="*80)
        return 0

    print(f"\n🤖 MODEL RESPONSE (Raw):")
    print(answer)
    
    # Try to extract JSON
    api_request = extract_json(answer)
    if not api_request:
        print(f"\n❌ EXTRACTED JSON: Invalid or missing JSON")
        print(f"⭐ REWARD SCORE: {format_score}")
        print("="*80)
        return format_score

    print(f"\n✅ EXTRACTED JSON:")
    print(json.dumps(api_request, indent=2))
    
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
    
    print(f"\n📊 SCORING BREAKDOWN:")
    for item in score_breakdown:
        print(f"   {item}")
    
    print(f"\n⭐ FINAL REWARD SCORE: {final_score}")
    print("="*80)
    
    return final_score 