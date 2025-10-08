# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def is_valid_sequence(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def count_search_turns(text: str) -> int:
    """Count the number of search turns in the solution string."""
    search_pattern = r"<search>(.*?)</search>"
    matches = re.findall(search_pattern, text, re.DOTALL)
    return len(matches)


def count_valid_search_turns(text: str, golden_answers: list[str]) -> int:
    """Count the number of valid search turns (where information contains correct answer)."""
    # Find all search-information pairs
    search_info_pattern = r"<search>(.*?)</search>\s*<information>(.*?)</information>"
    matches = re.findall(search_info_pattern, text, re.DOTALL)
    
    valid_count = 0
    for search_content, info_content in matches:
        # Check if this information block contains the correct answer
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(info_content):
                valid_count += 1
                break  # Found valid answer in this search, count it once
    
    return valid_count


def has_final_answer_tag(text: str) -> bool:
    """Check if the text ends with an answer tag (final turn requirement)."""
    # Find the last occurrence of answer tag
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, text, re.DOTALL))
    
    if not matches:
        return False
    
    # Check if there's any content after the last answer tag (excluding whitespace)
    last_match = matches[-1]
    remaining_text = text[last_match.end():].strip()
    
    # Should not have any significant content after the last answer tag
    return len(remaining_text) == 0


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_format(solution_str):
    is_valid_format, _ = is_valid_sequence(solution_str)
    if is_valid_format:
        return 1.0
    else:
        return 0.0


def compute_score_retrieval(solution_str, ground_truth):
    retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    if retrieval_correct:
        return 1.0
    else:
        return 0.0


def compute_score_em_format_retrievel(
    solution_str,
    ground_truth,
    method="strict",
    structure_format_score=0.2,
    final_format_score=0.1,
    retrieval_score=0.1,
    format_score=0,
    score=1.0,
):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return 0
    else:
        if em_check(answer, ground_truth['target']):
            if is_valid_format:
                return score # 1
            else:
                return score - structure_format_score # 0.8
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return final_format_score # 0.1


def compute_score_em_format_retrievel_with_search_penalty(
    solution_str,
    ground_truth,
    method="strict",
    structure_format_score=0.2,
    final_format_score=0.1,
    retrieval_score=0.1,
    format_score=0,
    score=1.0,
    search_penalty=0.1,
    min_search_turns=4,
    final_answer_penalty=0.3,
):
    """The scoring function for exact match (EM) with valid search turn penalty and final answer requirement.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        structure_format_score: the score for the structure format
        final_format_score: the score for the final format
        retrieval_score: the score for the retrieval
        format_score: the score for the format
        score: the score for the correct answer
        search_penalty: penalty per missing valid search turn (default 0.1)
        min_search_turns: minimum required valid search turns to avoid penalty (default 4)
        final_answer_penalty: penalty for not ending with answer tag (default 0.3)
    """
    # Get base score using existing function
    base_score = compute_score_em_format_retrievel(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        structure_format_score=structure_format_score,
        final_format_score=final_format_score,
        retrieval_score=retrieval_score,
        format_score=format_score,
        score=score,
    )
    
    # Count valid search turns and check final answer requirement
    valid_search_turns = count_valid_search_turns(solution_str, ground_truth['target'])
    has_final_answer = has_final_answer_tag(solution_str)
    
    # Get detailed scoring components for logging
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    answer_correct = False
    if answer is not None:
        answer_correct = em_check(answer, ground_truth['target'])
    
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Valid search turns: {valid_search_turns}")
        print(f"Min required valid search turns: {min_search_turns}")
        print(f"Has final answer tag: {has_final_answer}")
        print(f"Base score: {base_score}")
        print(f"Answer correct: {answer_correct}")
        print(f"Format valid: {is_valid_format}")
        print(f"Retrieval correct: {retrieval_correct}")
    
    # Apply penalties
    total_penalty = 0.0
    missing_valid_turns = 0
    valid_search_penalty_amount = 0.0
    
    # 1. Valid search penalty: 0.1 for each missing valid search turn below min_search_turns
    if valid_search_turns < min_search_turns:
        missing_valid_turns = min_search_turns - valid_search_turns
        valid_search_penalty_amount = missing_valid_turns * search_penalty
        total_penalty += valid_search_penalty_amount
        if do_print:
            print(f"Missing {missing_valid_turns} valid search turns, penalty: -{valid_search_penalty_amount:.2f}")
    
    # 2. Final answer penalty: 0.3 if not ending with answer tag
    if not has_final_answer:
        total_penalty += final_answer_penalty
        if do_print:
            print(f"Missing final answer tag, penalty: -{final_answer_penalty:.2f}")
    
    # Calculate final score
    final_score = base_score - total_penalty
    final_score = max(0.0, final_score)  # Ensure score doesn't go below 0
    
    if do_print:
        print(f"Total penalty: -{total_penalty:.2f}")
        print(f"Final score: {final_score}")
    
    # Store detailed metrics for wandb logging (will be accessed by the trainer)
    if not hasattr(compute_score_em_format_retrievel_with_search_penalty, 'detailed_metrics'):
        compute_score_em_format_retrievel_with_search_penalty.detailed_metrics = []
    
    compute_score_em_format_retrievel_with_search_penalty.detailed_metrics.append({
        # Keys expected by trainer
        'search_turns': valid_search_turns,  # Use valid_search_turns for the old search_turns key
        'missing_turns': missing_valid_turns,
        'base_score': base_score,
        'final_score': final_score,
        'penalty_applied': valid_search_turns < min_search_turns or not has_final_answer,
        'penalty_amount': total_penalty,
        'answer_correct': answer_correct,
        'format_valid': is_valid_format,
        'retrieval_correct': retrieval_correct,
        
        # Additional detailed keys for debugging
        'valid_search_turns': valid_search_turns,
        'min_search_turns': min_search_turns,
        'missing_valid_turns': missing_valid_turns,
        'has_final_answer': has_final_answer,
        'valid_search_penalty_applied': valid_search_turns < min_search_turns,
        'valid_search_penalty_amount': valid_search_penalty_amount,
        'final_answer_penalty_applied': not has_final_answer,
        'final_answer_penalty_amount': final_answer_penalty if not has_final_answer else 0.0,
        'total_penalty': total_penalty,
    })
    
    return final_score
