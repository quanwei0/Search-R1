#!/usr/bin/env python3
"""
简单的search penalty机制测试脚本
"""

import re

def count_search_turns(text: str) -> int:
    """Count the number of search turns in the solution string."""
    search_pattern = r"<search>(.*?)</search>"
    matches = re.findall(search_pattern, text, re.DOTALL)
    return len(matches)

def test_search_penalty():
    """测试search penalty功能"""
    
    # 测试用例1: 没有search的情况
    solution_no_search = """<|im_start|>assistant
<think>
我需要回答这个问题。
</think>

<answer>这是答案</answer>
"""
    
    # 测试用例2: 只有1次search的情况
    solution_one_search = """<|im_start|>assistant
<think>
我需要搜索一些信息。
</think>

<search>搜索查询1</search>

<information>搜索结果1</information>

<think>
现在我可以回答了。
</think>

<answer>这是答案</answer>
"""
    
    # 测试用例3: 有4次search的情况
    solution_four_search = """<|im_start|>assistant
<think>
我需要搜索一些信息。
</think>

<search>搜索查询1</search>

<information>搜索结果1</information>

<think>
需要更多信息。
</think>

<search>搜索查询2</search>

<information>搜索结果2</information>

<think>
还需要更多信息。
</think>

<search>搜索查询3</search>

<information>搜索结果3</information>

<think>
再搜索一次。
</think>

<search>搜索查询4</search>

<information>搜索结果4</information>

<think>
现在我可以回答了。
</think>

<answer>这是答案</answer>
"""
    
    print("=== 测试search turn计数功能 ===")
    count_no_search = count_search_turns(solution_no_search)
    count_one_search = count_search_turns(solution_one_search)
    count_four_search = count_search_turns(solution_four_search)
    
    print(f"没有search的情况: {count_no_search} 次search")
    print(f"1次search的情况: {count_one_search} 次search")
    print(f"4次search的情况: {count_four_search} 次search")
    
    print("\n=== 测试search penalty逻辑 ===")
    
    min_search_turns = 4
    search_penalty = 0.2
    base_score = 1.0  # 假设基础得分为1.0
    
    # 模拟penalty逻辑
    def apply_penalty(search_count, base_score, min_turns, penalty):
        if search_count < min_turns:
            return max(0.0, base_score - penalty)
        else:
            return base_score
    
    score_no_search = apply_penalty(count_no_search, base_score, min_search_turns, search_penalty)
    score_one_search = apply_penalty(count_one_search, base_score, min_search_turns, search_penalty)
    score_four_search = apply_penalty(count_four_search, base_score, min_search_turns, search_penalty)
    
    print(f"没有search的情况 - 得分: {score_no_search:.2f} (应该被扣除0.2分)")
    print(f"1次search的情况 - 得分: {score_one_search:.2f} (应该被扣除0.2分)")
    print(f"4次search的情况 - 得分: {score_four_search:.2f} (不应该被扣分)")
    
    print("\n=== 测试结果验证 ===")
    
    # 验证penalty是否正确应用
    if score_no_search < score_four_search:
        print("✅ 没有search的情况得分确实低于4次search的情况")
    else:
        print("❌ penalty机制可能有问题")
        
    if score_one_search < score_four_search:
        print("✅ 1次search的情况得分确实低于4次search的情况")
    else:
        print("❌ penalty机制可能有问题")
    
    # 验证penalty数值
    expected_penalty = 0.2
    actual_penalty = score_four_search - score_no_search
    if abs(actual_penalty - expected_penalty) < 0.01:
        print(f"✅ penalty数值正确: {actual_penalty:.2f}")
    else:
        print(f"❌ penalty数值可能不正确: 期望{expected_penalty}, 实际{actual_penalty:.2f}")
    
    # 验证计数功能
    if count_no_search == 0 and count_one_search == 1 and count_four_search == 4:
        print("✅ search计数功能正常")
    else:
        print(f"❌ search计数功能可能有问题: {count_no_search}, {count_one_search}, {count_four_search}")

if __name__ == "__main__":
    test_search_penalty()
