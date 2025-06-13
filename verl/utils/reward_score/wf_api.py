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


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.1, score=1.0
):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    expected_response = ground_truth
    print("Expected response from Jenish is ",expected_response)

    answer = extract_answer(solution_str)
    do_print = random.randint(1, 64) == 1
    # do_print = True

    if do_print:
        print("--------------------------------")
        print(f"Expected response: {expected_response}")
        print(f"Extracted answer: {answer}")
        print(f"LLM Response: {solution_str}")

    if answer is None:
        if do_print:
            print("No answer tags found. SCORE: 0")
        return 0

    # Check if valid JSON
    api_request = extract_json(answer)
    if not api_request:
        if do_print:
            print("Invalid JSON. SCORE: 0.1")
        return format_score

    # Check presence of keys: obj_code, fields, filters
    required_fields = ["obj_code", "fields", "filters"]
    field_presence_count = 0
    for field in required_fields:
        if field in api_request:
            field_presence_count += 1

    if field_presence_count < 3:
        print(f"Missing {3 - field_presence_count} fields")
        score = format_score + (field_presence_count * 0.1)
        return score

    # Check if obj_code is correct
    if api_request["obj_code"] != expected_response["obj_code"]:
        print(f"Incorrect obj_code: {api_request['obj_code']}")
        return 0.4

    # Check if filters are correct
    if sorted(api_request["filters"]) != sorted(expected_response["filters"]):
        print(f"Incorrect filters: {api_request['filters']}")
        return 0.5

    # Check if fields are correct
    for field in expected_response["fields"]:
        if field in ["ID", "name"]:
            continue

        if field not in api_request["fields"]:
            print(f"Missing field: {field}")
            return 0.6

    print("Correct API request")
    return 1.0
