import re
import random
import torch


def extract_solution(solution_str):
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[-1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[-1]

    answer_pattern = r'<answer>\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)\s*</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))

    if matches:
        match = matches[-1]
        return (match.group(1), match.group(2), match.group(3), match.group(4))
    else:
        return None


def validate_equation(equation, ground_truth):
    if len(equation) != len(ground_truth):
        return False

    if sum(map(int, equation[:3])) != int(equation[3]):
        return False

    sol_map = torch.full((26,), -1, dtype=torch.int)
    for num_str, enc_str in zip(equation, ground_truth):
        if len(num_str) != len(enc_str):
            return False

        for num_char, enc_char in zip(num_str, enc_str):
            num_val = int(num_char)
            enc_val = ord(enc_char) - ord('A')
            if sol_map[enc_val] == -1:
                sol_map[enc_val] = num_val
            elif sol_map[enc_val] != num_val:
                return False

    return True

def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    do_print = random.randint(0, 800) == 0
    do_print = False

    if do_print:
        print("--------------------------------")
        print(
            f"""Ground Truth : {solution_str} | {ground_truth[0]} + {ground_truth[1]} + {ground_truth[2]} = {ground_truth[3]}""")

    answer = extract_solution(solution_str)

    if answer is None:
        if do_print:
            print("Wrong Output Format")
        return 0.0

    try:
        if validate_equation(answer, ground_truth):
            if do_print:
                print(f"Correct Answer")
            return score
        else:
            if do_print:
                print("Wrong Answer, Correct Format")
            return format_score
    except:
        if do_print:
            print("Exception")
        return format_score