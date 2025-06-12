import json
import re
import random


def extract_api_request(solution_str):
    """Extract the API request from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split("\n")[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


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
    expected_response = ground_truth["expected_response"]

    api_request = extract_api_request(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Expected response: {expected_response}")
        print(f"Extracted API request: {api_request}")
        print(f"Solution string: {solution_str}")

    if api_request is None:
        if do_print:
            print("No API request found")
        return 0

    # Check if valid JSON
    if not is_valid_json(api_request):
        if do_print:
            print("Invalid JSON")
        return format_score

    # Check presence of keys: obj_code, fields, filters
    required_fields = ["obj_code", "fields", "filters"]
    missing_field_count = 0
    for field in required_fields:
        if field not in api_request:
            missing_field_count += 1
            print(f"Missing field: {field}")

    if missing_field_count > 0:
        score = 0.45 - (missing_field_count * 0.1)
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
    if sorted(api_request["fields"]) != sorted(expected_response["fields"]):
        print(f"Incorrect fields: {api_request['fields']}")
        return 0.6

    print("Correct API request")
    return 1.0
