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
    
    # if do_print:
    #     print(f"--------------------------------")
    #     print(f"Golden answers: {ground_truth['target']}")
    #     print(f"Extracted answer: {answer}")
    #     print(f"Solution string: {solution_str}")
    
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

    # if do_print:
    #     print(f"--------------------------------")
    #     print(f"Golden answers: {ground_truth['target']}")
    #     print(f"Extracted answer: {answer}")
    #     print(f"Solution string: {solution_str}")

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

###################################################################################################
def compute_score_final_em_format(final_turn_str, ground_truth):

    matches = list(re.finditer(r'<answer>(.*?)</answer>', final_turn_str, re.DOTALL))
    
    if len(matches) == 0:
        return 0.0
    answer = matches[0].group(1).strip()
    if em_check(answer, ground_truth['target']):
        return 1.0
    elif final_format_check(final_turn_str):
        return 0.2
    else:
        return 0.0


def final_format_check(final_turn_str: str) -> bool:
    content = final_turn_str

    if any(tag in content for tag in ["<search>", "</search>", "<information>", "</information>"]):
        return False

    if (len(re.findall(r"<think>", content))  != 1 or
        len(re.findall(r"</think>", content)) != 1 or
        len(re.findall(r"<answer>", content)) != 1 or
        len(re.findall(r"</answer>", content))!= 1):
        return False

    m_think_open  = re.search(r"<think>", content)
    m_think_close = re.search(r"</think>", content)
    m_ans_open    = re.search(r"<answer>", content)
    m_ans_close   = re.search(r"</answer>", content)

    if not (m_think_open.start()  < m_think_close.start() <
            m_ans_open.start()    < m_ans_close.start()):
        return False

    return True


def compute_score_step_retrieval_format(mid_turn_str, ground_truth):

    num_turn_minus_1 = len(mid_turn_str)
    step_rewards = []
    search_count = 0
    
    for i in range(num_turn_minus_1):
        turn_str = mid_turn_str[i]

        hit = is_retrieval_correct(turn_str, ground_truth['target'])
        retrieval_score = 0.3 if hit else 0.0

        format_score = 0.1 if mid_format_check(turn_str) else -0.2

        if "<search>" in turn_str:
            search_count += 1
            search_penalty = -0.1 * search_count
        else:
            search_penalty = 0.0
        
        r = retrieval_score + format_score + search_penalty
        step_rewards.append(r)
        
    return step_rewards


def mid_format_check(mid_turn_str):
    content = mid_turn_str

    for tag in ["think", "search", "information"]:
        open_count = len(re.findall(fr"<{tag}>", content))
        close_count = len(re.findall(fr"</{tag}>", content))
        if open_count != 1 or close_count != 1:
            return False

    think_open = re.search(r"<think>", content)
    search_open = re.search(r"<search>", content)
    info_open = re.search(r"<information>", content)

    if not (think_open and search_open and info_open):
        return False

    if not (think_open.start() < search_open.start() < info_open.start()):
        return False

    return True

###################################################################################################
def compute_score_f1(solution_str, ground_truth):

    ground_truths_list = list(ground_truth['target'])

    answer = set(extract_solution(solution_str=solution_str).strip().split())

    def f1(pred_tokens, gt_str):
        gt_tokens = set(gt_str.strip().split())
        IN = len(pred_tokens & gt_tokens)
        PN = len(pred_tokens)
        RN = len(gt_tokens)
        return 0.0 if PN + RN == 0 else 2 * IN / (PN + RN)

    max_f1 = max(f1(answer, gt) for gt in ground_truths_list)
    
    return max_f1