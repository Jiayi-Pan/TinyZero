import re
import random
import ast
import operator


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None

def validate_output_format(solution_str):
    """Validate the output format of the solution string."""
    # should contain only one <think> </think>, <answer> </answer>
    
    if "Assistant:" in solution_str:
        assitant_answer_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        assitant_answer_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return False
    
    think_count = assitant_answer_str.count('<think>')
    end_think_count = assitant_answer_str.count('</think>')
    answer_count = assitant_answer_str.count('<answer>')
    end_answer_count = assitant_answer_str.count('</answer>')
    # print(f'think_count: {think_count}, end_think_count: {end_think_count}, answer_count: {answer_count}, end_answer_count: {end_answer_count}')
    if think_count != end_think_count:
        return False
    if answer_count != end_answer_count:
        return False
    if answer_count != 1:
        return False
    return True

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
        
        equation_format_score = 0.05
        output_format_score = 0.05
        correct_score = 1.0
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1 # 对1/64的结果输出
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0
    
    equation_format_score = format_score * 0.5
    output_format_score = format_score * 0.5 if validate_output_format(solution_str) else 0
    final_format_score = equation_format_score + output_format_score
    if do_print:
        if output_format_score:
            print(f"Valid output format")
        else:
            print(f"Invalid output format")
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return final_format_score
    
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return final_format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            correct_answer_score = score * 0.9
            return correct_answer_score + final_format_score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return final_format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return final_format_score 