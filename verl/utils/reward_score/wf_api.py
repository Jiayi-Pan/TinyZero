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

    final_json_pattern = r"<final_json>(.*?)</final_json>"
    match = re.finditer(final_json_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) < 4:
        return None

    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def extract_text_after_thinking(solution_str):
    # Extract all text after the <thinking> tag
    thinking_start = solution_str.find("<thinking>")
    position_to_slice = thinking_start
    return solution_str[position_to_slice:].strip()


def strip_json_comments(json_str):
    """Remove // and /* */ style comments from JSON string."""
    # Remove single-line comments (// comment)
    json_str = re.sub(r"//.*?(?=\n|$)", "", json_str)

    # Remove multi-line comments (/* comment */)
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

    return json_str


def extract_json(answer):
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
                cleaned_json = strip_json_comments(json_str)
                return json.loads(cleaned_json)
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
    log_dir = os.path.expanduser(
        "/home/workfrontadmin/TinyZeroRL/workfront_training_logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"reward_logs_{datetime.datetime.now().strftime('%Y%m%d')}.txt"
    )

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

    log_both("\n" + "=" * 80)
    log_both("🤖 MODEL EVALUATION")
    log_both("=" * 80)

    # Show the question (extract from solution_str)
    question = None
    if "<|im_start|>user" in solution_str:
        # Qwen instruct format
        try:
            question = (
                solution_str.split("<|im_start|>user")[1].split("<|im_end|>")[0].strip()
            )
        except IndexError:
            question = "Could not extract question from Qwen format"
    elif "User:" in solution_str:
        # Standard format
        try:
            question = solution_str.split("User:")[-1].split("Assistant:")[0].strip()
        except IndexError:
            question = "Could not extract question from standard format"

    if question:
        log_both(f"📝 QUESTION: {question}")
    else:
        log_both("📝 QUESTION: Could not extract question from solution_str")

    log_both("\n🎯 EXPECTED RESPONSE:")
    log_both(json.dumps(expected_response, indent=2))
    log_both(f"🔍 RAW OUTPUT: {extract_text_after_thinking(solution_str)}")

    if answer is None:
        log_both("\n❌ No <final_json> tags found")
        log_both("⭐ REWARD SCORE: 0.0")
        log_both("=" * 80)
        return 0

    log_both("\n🤖 EXTRACTED RESPONSE (Raw):")
    log_both(answer)

    # Try to extract JSON
    api_request = extract_json(answer)
    if not api_request:
        log_both("\n❌ EXTRACTED JSON: Invalid or missing JSON")
        log_both(f"⭐ REWARD SCORE: {format_score}")
        log_both("=" * 80)
        return format_score

    # log_both(f"\n✅ EXTRACTED JSON:")
    # log_both(json.dumps(api_request, indent=2))

    # Detailed point breakdown system
    score_breakdown = []
    final_score = 0.0

    # Point allocation:
    # - Missing all 3 fields: -0.1 penalty
    # - objCode: 0.2 total (0.2 if correct, 0.05 if field exists but wrong)
    # - fields: 0.3 total (with partial credit)
    # - filters: 0.4 total (0.2 for correct filters + 0.2 for correct values, with partial credit)

    # Check for exact required field names
    required_fields = ["objCode", "fields", "filters"]
    missing_fields = []
    for field in required_fields:
        if field not in api_request:
            missing_fields.append(field)

    # Apply penalty if any exact field names are missing
    if missing_fields:
        final_score = format_score  # 0.1 penalty for missing exact field names
        score_breakdown.append(
            f" Missing exact required field names: {missing_fields} (-{1.0 - format_score:.1f} points, getting format_score {format_score})"
        )
        # Continue checking what's present for feedback, but score is capped at format_score
    else:
        score_breakdown.append(" All exact required field names present")

    # 1. objCode scoring (0.2 total points)
    if "objCode" in api_request:
        if api_request["objCode"] == expected_response["objCode"]:
            final_score += 0.2
            score_breakdown.append(
                f"objCode: Correct '{api_request['objCode']}' (+0.2 points)"
            )
        else:
            final_score += 0.05  # Partial credit for field existing
            score_breakdown.append(
                f"objCode: Wrong - got '{api_request['objCode']}', expected '{expected_response['objCode']}' (+0.05 points for field existing)"
            )

    # 2. fields scoring (0.3 total points)
    if "fields" in api_request:
        model_fields = api_request.get("fields", [])
        if model_fields:
            model_fields = set(model_fields)
        expected_fields = set(expected_response.get("fields", []))

        if not model_fields and expected_fields:
            score_breakdown.append(" fields: Empty but expected content (0.0 points)")
        elif not expected_fields:
            final_score += 0.3  # Full credit if no specific fields expected
            score_breakdown.append(" fields: No specific requirements (+0.3 points)")
        else:
            # Allow some flexibility for ID and name fields
            core_expected = expected_fields - {"ID", "name"}
            core_model = model_fields - {"ID", "name"}

            if not core_expected:
                # No core fields expected, check if any expected fields present
                matching_fields = len(expected_fields.intersection(model_fields))
                total_expected = len(expected_fields)
                field_score = (matching_fields / total_expected) * 0.3
                final_score += field_score
                score_breakdown.append(
                    f" fields: {matching_fields}/{total_expected} expected fields present (+{field_score:.2f} points)"
                )
            else:
                # Check core fields
                matching_core = len(core_expected.intersection(core_model))
                total_core = len(core_expected)

                if matching_core == total_core:
                    final_score += 0.3
                    score_breakdown.append(
                        " fields: All important fields present (+0.3 points)"
                    )
                else:
                    # Partial credit for fields
                    field_score = (matching_core / total_core) * 0.3
                    final_score += field_score
                    missing = core_expected - core_model
                    score_breakdown.append(
                        f"⚠️ fields: Partial match {matching_core}/{total_core}, missing {missing} (+{field_score:.2f} points)"
                    )

    # 3. filters scoring (0.4 total points: 0.2 for correct filters + 0.2 for correct values)
    if "filters" in api_request:
        model_filters = api_request.get("filters", {})
        expected_filters = expected_response.get("filters", {})

        if not model_filters and expected_filters:
            score_breakdown.append(" filters: Empty but expected content (0.0 points)")
        elif not expected_filters:
            final_score += 0.4  # Full credit if no filters expected
            score_breakdown.append("✅ filters: No specific requirements (+0.4 points)")
        elif isinstance(model_filters, dict) and isinstance(expected_filters, dict):
            # Check filter keys (0.2 points)
            expected_keys = set(expected_filters.keys())
            model_keys = set(model_filters.keys())

            matching_keys = len(expected_keys.intersection(model_keys))
            total_keys = len(expected_keys)

            if matching_keys == total_keys:
                key_score = 0.2
                score_breakdown.append("filters: All filter keys present (+0.2 points)")
            else:
                key_score = (matching_keys / total_keys) * 0.2 if total_keys > 0 else 0
                missing_keys = expected_keys - model_keys
                score_breakdown.append(
                    f" filters: Partial keys {matching_keys}/{total_keys}, missing {missing_keys} (+{key_score:.2f} points)"
                )

            final_score += key_score

            # Check filter values (0.2 points)
            matching_values = sum(
                1
                for k, v in expected_filters.items()
                if k in model_filters and model_filters[k] == v
            )

            if matching_values == total_keys:
                value_score = 0.2
                score_breakdown.append(
                    "filters: All filter values correct (+0.2 points)"
                )
            else:
                value_score = (
                    (matching_values / total_keys) * 0.2 if total_keys > 0 else 0
                )
                score_breakdown.append(
                    f" filters: Partial values {matching_values}/{total_keys} correct (+{value_score:.2f} points)"
                )

            final_score += value_score
        else:
            score_breakdown.append(" filters: Format mismatch (0.0 points)")

    # Ensure score doesn't exceed 1.0 or go below 0
    # If missing field names, cap at format_score
    if missing_fields:
        final_score = min(final_score, format_score)
    final_score = max(0, min(final_score, 1.0))

    final_score = round(final_score, 2)
    log_both(f"\n📊 SCORING BREAKDOWN:")
    for item in score_breakdown:
        log_both(f"   {item}")

    log_both(f"\n⭐ FINAL REWARD SCORE: {final_score}")
    log_both("=" * 80)

    return final_score
