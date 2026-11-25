"""Checks for display heuristics, independent of model/API access."""

import unittest

from code_quality import extract_code_from_response, evaluate_code_quality


class CodeQualityTests(unittest.TestCase):
    def test_short_raw_python(self):
        for code in ['print(1)', 'x = 1', 'for i in range(2):\n    print(i)']:
            with self.subTest(code=code):
                self.assertEqual(extract_code_from_response(code), code)
                score, metrics = evaluate_code_quality(code)
                self.assertEqual(score, 1.0)
                self.assertEqual(metrics, {'syntax': 'Valid', 'density': 1.0})

    def test_fenced_code_with_explanation(self):
        response = 'Suggested program:\n```python\nprint(1)\n```\nReview before use.'
        self.assertEqual(extract_code_from_response(response), 'print(1)')
        score, metrics = evaluate_code_quality(response)
        self.assertEqual(metrics['syntax'], 'Valid')
        self.assertLess(metrics['density'], 1.0)
        self.assertGreater(score, 0.6)
        self.assertLess(score, 1.0)

    def test_unlabelled_and_uppercase_fences(self):
        for label in ['', 'PYTHON']:
            self.assertEqual(extract_code_from_response(f'```{label}\nprint(1)\n```'), 'print(1)')

    def test_empty_and_non_code(self):
        for response in ['', '   ', 'Here is the corrected program:']:
            self.assertEqual(extract_code_from_response(response), '')
            self.assertEqual(evaluate_code_quality(response)[0], 0.0)

    def test_fenced_syntax_error_has_no_syntax_credit(self):
        score, metrics = evaluate_code_quality('```python\nprint(\n```')
        self.assertEqual(metrics['syntax'], 'Syntax error')
        self.assertLessEqual(score, 0.4)

    def test_runtime_failure_can_still_score_highly(self):
        # The heuristic parses code but cannot establish runtime success or a fix.
        score, metrics = evaluate_code_quality('print(1 / 0)')
        self.assertEqual(score, 1.0)
        self.assertEqual(metrics['syntax'], 'Valid')


if __name__ == '__main__':
    unittest.main()
