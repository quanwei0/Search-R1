import json
import os
import re
import string
import random
import sys
import glob

from tabulate import tabulate


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
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth):

    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth["target"]):
            return 1
        else:
            return 0


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
            return (
                False,
                f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags",
            )

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
                # if part.strip():
                #     return (
                #         False,
                #         f"Unexpected content '{part.strip()}' between tags (state: {state})",
                #     )
                pass
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
    retrieval_correct = is_retrieval_correct(solution_str, ground_truth["target"])
    if retrieval_correct:
        return 1.0
    else:
        return 0.0

def compute_score_subem(solution_str, ground_truth):
    answer = extract_solution(solution_str=solution_str)
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return 1
        else:
            return 0


def compute_score_f1(solution_str, ground_truth):
    answer = extract_solution(solution_str=solution_str)
    
    if answer is None:
        return 0.0
    
    ground_truths_list = ground_truth['target'] if isinstance(ground_truth['target'], list) else [ground_truth['target']]
    pred_tokens = set(answer.strip().split())

    def f1(pred_tokens, gt_str):
        gt_tokens = set(gt_str.strip().split())
        IN = len(pred_tokens & gt_tokens)
        PN = len(pred_tokens)
        RN = len(gt_tokens)
        return 0.0 if PN + RN == 0 else 2 * IN / (PN + RN)

    max_f1 = max(f1(pred_tokens, gt) for gt in ground_truths_list)
    return max_f1


def compute_num_turns(trajectory):
    """
    Compute the number of turns in a trajectory.
    A turn is represented by each element in the turn_texts array.
    """
    turn_texts = trajectory.get("turn_texts", [])
    return len(turn_texts)


def test_trajectory_comprehensive(json_file_path):
    """
    Read trajectory JSON file and compute accuracy, format, and retrieval metrics
    """

    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} does not exist")
        return

    # Read JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:
        trajectories = json.load(f)

    total_samples = len(trajectories)
    accuracy_correct = 0
    format_correct = 0
    retrieval_correct = 0
    subem_correct = 0
    f1_scores = 0
    total_num_turns = 0

    # Track metrics by data source
    data_source_stats = {}

    print(f"Total samples: {total_samples}")
    print("=" * 60)

    for i, trajectory in enumerate(trajectories):
        # Get generated text and ground truth labels
        full_text = trajectory.get("full_text", "")
        ground_truth = trajectory.get("ground_truth", [])
        data_source = trajectory.get("data_source", "unknown")

        # Construct ground_truth dict format for compute_score_em
        gt_dict = {"target": ground_truth}

        # Calculate all scores
        acc_score = compute_score_em(full_text, gt_dict)
        format_score = compute_score_format(full_text)
        retrieval_score = compute_score_retrieval(full_text, gt_dict)
        subem_score = compute_score_subem(full_text, gt_dict)
        f1_score = compute_score_f1(full_text, gt_dict)
        num_turns = compute_num_turns(trajectory)

        accuracy_correct += acc_score
        format_correct += format_score
        retrieval_correct += retrieval_score
        subem_correct += subem_score
        f1_scores += f1_score
        total_num_turns += num_turns

        # Track by data source
        if data_source not in data_source_stats:
            data_source_stats[data_source] = {
                "accuracy_correct": 0,
                "format_correct": 0,
                "retrieval_correct": 0,
                "subem_correct": 0,
                "f1_scores": 0,
                "num_turns": 0,
                "total": 0,
            }
        data_source_stats[data_source]["accuracy_correct"] += acc_score
        data_source_stats[data_source]["format_correct"] += format_score
        data_source_stats[data_source]["retrieval_correct"] += retrieval_score
        data_source_stats[data_source]["subem_correct"] += subem_score
        data_source_stats[data_source]["f1_scores"] += f1_score
        data_source_stats[data_source]["num_turns"] += num_turns
        data_source_stats[data_source]["total"] += 1

        # Calculate current metrics
        current_accuracy = accuracy_correct / (i + 1)
        current_format = format_correct / (i + 1)
        current_retrieval = retrieval_correct / (i + 1)
        current_subem = subem_correct / (i + 1)
        current_f1 = f1_scores / (i + 1)
        current_avg_turns = total_num_turns / (i + 1)

        # Stream print current metrics
        print(
            f"\rProgress: {i+1}/{total_samples} | Acc: {current_accuracy:.4f} | Format: {current_format:.4f} | Retrieval: {current_retrieval:.4f} | SubEM: {current_subem:.4f} | F1: {current_f1:.4f} | Avg Turns: {current_avg_turns:.2f}",
            end="",
            flush=True,
        )

    # Print final results
    print(f"\n\nFINAL RESULTS:")
    print("=" * 80)
    final_accuracy = accuracy_correct / total_samples
    final_format = format_correct / total_samples
    final_retrieval = retrieval_correct / total_samples
    final_subem = subem_correct / total_samples
    final_f1 = f1_scores / total_samples
    final_avg_turns = total_num_turns / total_samples

    print(
        f"Accuracy:  {int(accuracy_correct):4d}/{total_samples:4d} = {final_accuracy:.4f} ({final_accuracy*100:.2f}%)"
    )
    print(
        f"Format:    {int(format_correct):4d}/{total_samples:4d} = {final_format:.4f} ({final_format*100:.2f}%)"
    )
    print(
        f"Retrieval: {int(retrieval_correct):4d}/{total_samples:4d} = {final_retrieval:.4f} ({final_retrieval*100:.2f}%)"
    )
    print(
        f"SubEM:     {int(subem_correct):4d}/{total_samples:4d} = {final_subem:.4f} ({final_subem*100:.2f}%)"
    )
    print(
        f"F1:        {f1_scores:8.2f}/{total_samples:4d} = {final_f1:.4f}"
    )
    print(
        f"Avg Turns: {total_num_turns:8.0f}/{total_samples:4d} = {final_avg_turns:.2f}"
    )

    # Print metrics by data source
    print(f"\nMETRICS BY DATA SOURCE:")
    table_data = []
    for source, stats in sorted(data_source_stats.items()):
        if stats["total"] > 0:
            acc_pct = (stats["accuracy_correct"] / stats["total"]) * 100
            fmt_pct = (stats["format_correct"] / stats["total"]) * 100
            ret_pct = (stats["retrieval_correct"] / stats["total"]) * 100
            subem_pct = (stats["subem_correct"] / stats["total"]) * 100
            f1_avg = stats["f1_scores"] / stats["total"]
            turns_avg = stats["num_turns"] / stats["total"]
            table_data.append([
                source,
                f"{acc_pct:.2f}%",
                f"{fmt_pct:.2f}%", 
                f"{ret_pct:.2f}%",
                f"{subem_pct:.2f}%",
                f"{f1_avg:.4f}",
                f"{turns_avg:.2f}"
            ])
    
    headers = ["Source", "Accuracy", "Format", "Retrieval", "SubEM", "F1", "Avg Turns"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    return final_accuracy, final_format, final_retrieval, final_subem, final_f1, final_avg_turns


def test_directory_comprehensive(directory_path):
    """
    Read all JSON files in directory and compute combined accuracy, format, and retrieval metrics
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist")
        return

    # Find all JSON files in directory
    json_files = glob.glob(os.path.join(directory_path, "*.json"))

    # Sort files by the batch number extracted from filename
    def extract_batch_number(filename):
        import re

        match = re.search(r"batch_(\d+)", filename)
        return int(match.group(1)) if match else 0

    json_files.sort(key=extract_batch_number)  # Sort by batch number

    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return

    print(f"Found {len(json_files)} JSON files")
    print(f"Directory: {directory_path}")
    print("=" * 60)

    # Combined statistics
    total_samples = 0
    total_accuracy_correct = 0
    total_format_correct = 0
    total_retrieval_correct = 0
    total_subem_correct = 0
    total_f1_scores = 0
    total_num_turns = 0
    combined_data_source_stats = {}

    for file_idx, json_file in enumerate(json_files):
        print(
            f"\nProcessing file {file_idx + 1}/{len(json_files)}: {os.path.basename(json_file)}"
        )

        # Read JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            trajectories = json.load(f)

        file_samples = len(trajectories)
        file_accuracy_correct = 0
        file_format_correct = 0
        file_retrieval_correct = 0
        file_subem_correct = 0
        file_f1_scores = 0
        file_num_turns = 0

        for i, trajectory in enumerate(trajectories):
            # Get generated text and ground truth labels
            full_text = trajectory.get("full_text", "")
            ground_truth = trajectory.get("ground_truth", [])
            data_source = trajectory.get("data_source", "unknown")

            # Construct ground_truth dict format for compute_score_em
            gt_dict = {"target": ground_truth}

            # Calculate all scores
            acc_score = compute_score_em(full_text, gt_dict)
            format_score = compute_score_format(full_text)
            retrieval_score = compute_score_retrieval(full_text, gt_dict)
            subem_score = compute_score_subem(full_text, gt_dict)
            f1_score = compute_score_f1(full_text, gt_dict)
            num_turns = compute_num_turns(trajectory)

            file_accuracy_correct += acc_score
            file_format_correct += format_score
            file_retrieval_correct += retrieval_score
            file_subem_correct += subem_score
            file_f1_scores += f1_score
            file_num_turns += num_turns
            total_accuracy_correct += acc_score
            total_format_correct += format_score
            total_retrieval_correct += retrieval_score
            total_subem_correct += subem_score
            total_f1_scores += f1_score
            total_num_turns += num_turns

            # Track by data source
            if data_source not in combined_data_source_stats:
                combined_data_source_stats[data_source] = {
                    "accuracy_correct": 0,
                    "format_correct": 0,
                    "retrieval_correct": 0,
                    "subem_correct": 0,
                    "f1_scores": 0,
                    "num_turns": 0,
                    "total": 0,
                }
            combined_data_source_stats[data_source]["accuracy_correct"] += acc_score
            combined_data_source_stats[data_source]["format_correct"] += format_score
            combined_data_source_stats[data_source][
                "retrieval_correct"
            ] += retrieval_score
            combined_data_source_stats[data_source]["subem_correct"] += subem_score
            combined_data_source_stats[data_source]["f1_scores"] += f1_score
            combined_data_source_stats[data_source]["num_turns"] += num_turns
            combined_data_source_stats[data_source]["total"] += 1

        total_samples += file_samples
        file_accuracy = file_accuracy_correct / file_samples if file_samples > 0 else 0
        file_format = file_format_correct / file_samples if file_samples > 0 else 0
        file_retrieval = (
            file_retrieval_correct / file_samples if file_samples > 0 else 0
        )
        file_subem = file_subem_correct / file_samples if file_samples > 0 else 0
        file_f1 = file_f1_scores / file_samples if file_samples > 0 else 0
        file_avg_turns = file_num_turns / file_samples if file_samples > 0 else 0
        print(
            f"File {os.path.basename(json_file)}: Acc {file_accuracy:.4f} | Format {file_format:.4f} | Retrieval {file_retrieval:.4f} | SubEM {file_subem:.4f} | F1 {file_f1:.4f} | Avg Turns {file_avg_turns:.2f}"
        )

    # Calculate overall metrics
    overall_accuracy = (
        total_accuracy_correct / total_samples if total_samples > 0 else 0
    )
    overall_format = total_format_correct / total_samples if total_samples > 0 else 0
    overall_retrieval = (
        total_retrieval_correct / total_samples if total_samples > 0 else 0
    )
    overall_subem = total_subem_correct / total_samples if total_samples > 0 else 0
    overall_f1 = total_f1_scores / total_samples if total_samples > 0 else 0
    overall_avg_turns = total_num_turns / total_samples if total_samples > 0 else 0

    print(f"\n" + "=" * 90)
    print(f"COMBINED RESULTS:")
    print(f"Total files processed: {len(json_files)}")
    print(f"Total samples: {total_samples}")
    print(
        f"Accuracy:  {int(total_accuracy_correct):4d}/{total_samples:4d} = {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)"
    )
    print(
        f"Format:    {int(total_format_correct):4d}/{total_samples:4d} = {overall_format:.4f} ({overall_format*100:.2f}%)"
    )
    print(
        f"Retrieval: {int(total_retrieval_correct):4d}/{total_samples:4d} = {overall_retrieval:.4f} ({overall_retrieval*100:.2f}%)"
    )
    print(
        f"SubEM:     {int(total_subem_correct):4d}/{total_samples:4d} = {overall_subem:.4f} ({overall_subem*100:.2f}%)"
    )
    print(
        f"F1:        {total_f1_scores:8.2f}/{total_samples:4d} = {overall_f1:.4f}"
    )
    print(
        f"Avg Turns: {total_num_turns:8.0f}/{total_samples:4d} = {overall_avg_turns:.2f}"
    )

    # Print metrics by data source
    print(f"\nMETRICS BY DATA SOURCE:")
    table_data = []
    for source, stats in sorted(combined_data_source_stats.items()):
        if stats["total"] > 0:
            acc_pct = (stats["accuracy_correct"] / stats["total"]) * 100
            fmt_pct = (stats["format_correct"] / stats["total"]) * 100
            ret_pct = (stats["retrieval_correct"] / stats["total"]) * 100
            subem_pct = (stats["subem_correct"] / stats["total"]) * 100
            f1_avg = stats["f1_scores"] / stats["total"]
            turns_avg = stats["num_turns"] / stats["total"]
            table_data.append([
                source,
                f"{acc_pct:.2f}%",
                f"{fmt_pct:.2f}%", 
                f"{ret_pct:.2f}%",
                f"{subem_pct:.2f}%",
                f"{f1_avg:.4f}",
                f"{turns_avg:.2f}"
            ])
    
    headers = ["Source", "Accuracy", "Format", "Retrieval", "SubEM", "F1", "Avg Turns"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


    return overall_accuracy, overall_format, overall_retrieval, overall_subem, overall_f1, overall_avg_turns


if __name__ == "__main__":

    # Directory containing all JSON files
    directory_path = (
        "./outputs/log_val_traj/qw-val-nq-ppo-mixed-reward-qwen2.5-7b-maxturn4-step500-20251009-023312_20251009_024249"
    )

    # Use command line argument if provided
    if len(sys.argv) > 1:
        path_arg = sys.argv[1]
        if os.path.isdir(path_arg):
            print(f"Testing directory with comprehensive metrics: {path_arg}")
            test_directory_comprehensive(path_arg)
        elif os.path.isfile(path_arg):
            print(f"Testing single file with comprehensive metrics: {path_arg}")
            test_trajectory_comprehensive(path_arg)
        else:
            print(f"Error: Path {path_arg} does not exist")
    else:
        print(f"Testing directory with comprehensive metrics: {directory_path}")
        test_directory_comprehensive(directory_path)
