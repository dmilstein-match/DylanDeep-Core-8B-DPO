"""
Comprehensive unit tests for answer extraction and normalization utilities.

Tests ensure consistent behavior across:
- Different answer formats (Answer:, ####, plain text)
- Edge cases (no numbers, empty strings, malformed input)
- Normalization (whitespace, case, commas, decimals)
"""

import pytest
from src.common.answer_utils import extract_answer, normalize_answer


class TestExtractAnswer:
    """Test suite for extract_answer() function."""

    # Test "Answer:" format (preferred)
    def test_extract_with_answer_marker(self):
        text = "Let me solve this step by step. Answer: 42"
        assert extract_answer(text) == "42"

    def test_extract_with_answer_marker_negative(self):
        text = "The result is Answer: -17"
        assert extract_answer(text) == "-17"

    def test_extract_with_answer_marker_decimal(self):
        text = "Therefore Answer: 3.14159"
        assert extract_answer(text) == "3.14159"

    def test_extract_with_answer_marker_multiple_numbers_after(self):
        text = "Answer: The answer is 42 but also 17"
        # Should take the first number after "Answer:"
        assert extract_answer(text) == "42"

    def test_extract_with_multiple_answer_markers(self):
        text = "Answer: 10 is wrong. Answer: 42"
        # Should use the last "Answer:" marker
        assert extract_answer(text) == "42"

    # Test "####" format (backward compatibility)
    def test_extract_with_quad_hash_marker(self):
        text = "Step by step solution. #### 42"
        assert extract_answer(text) == "42"

    def test_extract_with_quad_hash_negative(self):
        text = "The balance is #### -100"
        assert extract_answer(text) == "-100"

    def test_extract_with_quad_hash_decimal(self):
        text = "Final answer #### 2.5"
        assert extract_answer(text) == "2.5"

    def test_extract_prefers_answer_over_quad_hash(self):
        text = "First #### 10, but Answer: 42"
        # Should prefer "Answer:" over "####"
        assert extract_answer(text) == "42"

    # Test fallback to last number
    def test_extract_last_number_no_markers(self):
        text = "There are 10 apples and 15 oranges, total is 25"
        assert extract_answer(text) == "25"

    def test_extract_last_number_with_text_after(self):
        text = "The answer is 42 items"
        assert extract_answer(text) == "42"

    def test_extract_last_negative_number(self):
        text = "The deficit is -250"
        assert extract_answer(text) == "-250"

    def test_extract_last_decimal(self):
        text = "The percentage is 87.5"
        assert extract_answer(text) == "87.5"

    # Test edge cases
    def test_extract_no_numbers(self):
        text = "No numbers here"
        # Should return the text as-is
        assert extract_answer(text) == "No numbers here"

    def test_extract_empty_string(self):
        text = ""
        assert extract_answer(text) == ""

    def test_extract_whitespace_only(self):
        text = "   "
        assert extract_answer(text) == ""

    def test_extract_single_number(self):
        text = "42"
        assert extract_answer(text) == "42"

    def test_extract_number_with_leading_zero(self):
        text = "The code is 007"
        assert extract_answer(text) == "007"

    def test_extract_large_number(self):
        text = "The population is 1234567890"
        assert extract_answer(text) == "1234567890"

    def test_extract_decimal_without_leading_digit(self):
        text = "The value is .5"
        assert extract_answer(text) == ".5"

    def test_extract_zero(self):
        text = "Answer: 0"
        assert extract_answer(text) == "0"

    def test_extract_negative_decimal(self):
        text = "Answer: -3.14"
        assert extract_answer(text) == "-3.14"

    # Test with realistic math problem outputs
    def test_extract_from_math_solution(self):
        text = """Let me solve this step by step:
        1) First, calculate 5 + 3 = 8
        2) Then multiply by 2: 8 × 2 = 16
        3) Finally subtract 4: 16 - 4 = 12

        Answer: 12"""
        assert extract_answer(text) == "12"

    def test_extract_from_gsm8k_format(self):
        text = """Step 1: Calculate initial amount
        Step 2: Add the increase
        Step 3: Subtract the decrease
        #### 42"""
        assert extract_answer(text) == "42"


class TestNormalizeAnswer:
    """Test suite for normalize_answer() function."""

    # Test whitespace handling
    def test_normalize_strips_whitespace(self):
        assert normalize_answer("  42  ") == "42"

    def test_normalize_strips_newlines(self):
        assert normalize_answer("42\n") == "42"

    def test_normalize_strips_tabs(self):
        assert normalize_answer("\t42\t") == "42"

    # Test case handling
    def test_normalize_lowercases(self):
        assert normalize_answer("ABC") == "abc"

    def test_normalize_mixed_case(self):
        assert normalize_answer("AbC123") == "abc123"

    # Test comma removal
    def test_normalize_removes_comma_in_thousands(self):
        assert normalize_answer("1,000") == "1000"

    def test_normalize_removes_multiple_commas(self):
        assert normalize_answer("1,234,567") == "1234567"

    def test_normalize_removes_comma_in_text(self):
        assert normalize_answer("hello,world") == "helloworld"

    # Test decimal normalization
    def test_normalize_whole_number_decimal(self):
        # 42.0 should become 42
        assert normalize_answer("42.0") == "42"

    def test_normalize_whole_number_with_zeros(self):
        # 100.00 should become 100
        assert normalize_answer("100.00") == "100"

    def test_normalize_preserves_non_whole_decimal(self):
        # 42.5 should stay as 42.5
        assert normalize_answer("42.5") == "42.5"

    def test_normalize_preserves_precision(self):
        # 3.14159 should stay as 3.14159
        assert normalize_answer("3.14159") == "3.14159"

    def test_normalize_negative_whole_decimal(self):
        # -42.0 should become -42
        assert normalize_answer("-42.0") == "-42"

    def test_normalize_negative_non_whole_decimal(self):
        # -42.5 should stay as -42.5
        assert normalize_answer("-42.5") == "-42.5"

    # Test edge cases
    def test_normalize_empty_string(self):
        assert normalize_answer("") == ""

    def test_normalize_whitespace_only(self):
        assert normalize_answer("   ") == ""

    def test_normalize_single_number(self):
        assert normalize_answer("42") == "42"

    def test_normalize_zero(self):
        assert normalize_answer("0") == "0"

    def test_normalize_zero_decimal(self):
        assert normalize_answer("0.0") == "0"

    def test_normalize_non_numeric_text(self):
        # Should pass through without error
        assert normalize_answer("hello") == "hello"

    def test_normalize_mixed_text_and_numbers(self):
        assert normalize_answer("42 apples") == "42 apples"

    def test_normalize_invalid_decimal(self):
        # Should not crash on invalid decimals
        ans = normalize_answer("12.34.56")
        assert ans == "12.34.56"

    # Test combined operations
    def test_normalize_combined_whitespace_comma_decimal(self):
        # "  1,000.0  " -> "1000"
        assert normalize_answer("  1,000.0  ") == "1000"

    def test_normalize_combined_case_comma(self):
        # "ABC,123" -> "abc123"
        assert normalize_answer("ABC,123") == "abc123"

    def test_normalize_scientific_notation_converted(self):
        # Python's float() actually handles scientific notation correctly
        ans = normalize_answer("1.5e10")
        # 1.5e10 = 15000000000 (whole number, so .0 is removed)
        assert ans == "15000000000"


class TestAnswerExtractAndNormalizeWorkflow:
    """Test the full workflow of extract then normalize."""

    def test_workflow_answer_marker(self):
        text = "The solution is Answer: 1,000.0"
        extracted = extract_answer(text)
        normalized = normalize_answer(extracted)
        assert normalized == "1000"

    def test_workflow_quad_hash_marker(self):
        text = "Step by step #### 42.00"
        extracted = extract_answer(text)
        normalized = normalize_answer(extracted)
        assert normalized == "42"

    def test_workflow_no_marker(self):
        text = "There are 2,500 items total"
        extracted = extract_answer(text)
        normalized = normalize_answer(extracted)
        assert normalized == "2500"

    def test_workflow_correctness_detection(self):
        # Simulate checking if model answer matches gold answer
        model_output = "Let me calculate: Answer: 42.0"
        gold_answer = "#### 42"

        model_ans = normalize_answer(extract_answer(model_output))
        gold_ans = normalize_answer(extract_answer(gold_answer))

        assert model_ans == gold_ans == "42"

    def test_workflow_different_formats_match(self):
        # Different formats should match after extraction + normalization
        outputs = [
            "Answer: 1,000.0",
            "#### 1000",
            "The answer is 1,000",
            "Total: 1000.00"
        ]

        normalized_answers = [
            normalize_answer(extract_answer(out)) for out in outputs
        ]

        # All should normalize to "1000"
        assert all(ans == "1000" for ans in normalized_answers)


class TestRealWorldScenarios:
    """Test scenarios from actual model outputs and datasets."""

    def test_gsm8k_format_gold_answer(self):
        # Typical GSM8K gold answer format
        gold = "Janet sells 16 - 3 - 4 = 9 duck eggs a day. #### 9"
        assert normalize_answer(extract_answer(gold)) == "9"

    def test_model_output_with_reasoning(self):
        # Model output with chain of thought
        output = """Let me solve this problem step by step.

        First, I need to calculate the total cost.
        10 items at $5 each = 10 × 5 = 50

        Then apply the discount of $8.
        50 - 8 = 42

        Answer: 42"""
        assert normalize_answer(extract_answer(output)) == "42"

    def test_model_output_with_units(self):
        # Model sometimes includes units
        output = "The total distance is Answer: 250 miles"
        # Extract just gets the first number after Answer:
        assert extract_answer(output) == "250"

    def test_negative_answer_debt_problem(self):
        output = "After losing money, the balance is Answer: -150"
        assert normalize_answer(extract_answer(output)) == "-150"

    def test_decimal_answer_percentage_problem(self):
        output = "The percentage is #### 87.5"
        assert normalize_answer(extract_answer(output)) == "87.5"

    def test_large_number_with_commas(self):
        gold = "The population increased to #### 1,234,567"
        assert normalize_answer(extract_answer(gold)) == "1234567"

    def test_zero_answer(self):
        output = "All items cancel out. Answer: 0"
        assert normalize_answer(extract_answer(output)) == "0"

    def test_answer_with_extra_text_after_marker(self):
        # Sometimes models add explanation after the answer
        output = "Answer: 42 (this is because we multiplied 6 by 7)"
        assert extract_answer(output) == "42"
