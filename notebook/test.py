import unittest
from Translate import Translate
import warnings
warnings.filterwarnings("ignore")

class TestTranslate(unittest.TestCase):
    def test_translator(self):
        # Setup
        model_name = f"mbart-finetuned-cn-to-en-auto"
        model_path = f"../models/{model_name}"
        trans = Translate(model_path)

        # Test input
        text = "开空调的情况下，续航掉的太快了，特别是冬天天气冷的时候，不开空调不行，天气一冻，续航就掉的更快"

        # Expected output
        # expected_output = "With the air conditioner on, the battery drains too quickly. Especially in the winter when the weather is cold, you can't do without the air conditioner. As soon as the weather gets cold, the battery drains even faster."

        # Actual output
        actual_output = trans.translator(text)

        # Assertion
        # self.assertEqual(expected_output, actual_output)
        print("-" * 70)
        print("TEXT: " + text)
        print("TRANSLATION: " + actual_output)
        

if __name__ == "__main__":
    unittest.main()
