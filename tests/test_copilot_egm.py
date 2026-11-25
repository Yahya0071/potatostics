"""
tests/test_copilot_egm.py - Unit Tests for Phase 7: Elephant-Goldfish Dual-Memory AI
1. Goldfish Memory Buffer (Sliding Window Context, Turn Recording)
2. Elephant Long-Term Persistent Memory (Codebooks, Project Preferences)
3. Contextual Reasoning:
   - Anaphora / pronoun resolution ("what was the R2?")
   - Contextual modification ("add robust standard errors")
   - Automated econometric coefficient interpretation ("explain that")
4. Causal & Export Prompt Translation (csdid, eventstudy, esttab)
5. Memory inspection & codebook commands (ai memory, ai codebook)
"""

import os
import unittest
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.copilot_egm import GoldfishMemoryBuffer, ElephantLongTermMemory, ElephantGoldfishCopilot


class TestCopilotEGM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(os.path.dirname(__file__), "test_egm_outputs")
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.memory_file = os.path.join(cls.test_dir, "test_elephant.json")

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(20, 5, n)
        x2 = np.random.normal(3000, 500, n)
        y = 1000 + 150 * x1 - 0.5 * x2 + np.random.normal(0, 50, n)

        self.df = pd.DataFrame({
            'price': y,
            'mpg': x1,
            'weight': x2,
            'unit': np.random.choice([1, 2, 3, 4], size=n),
            'time': np.random.choice([2010, 2011, 2012, 2013], size=n),
            'cohort': np.random.choice([2011, 2012, 0], size=n)
        })

        self.dataset = Dataset()
        self.dataset.load_dataframe(self.df)
        self.interp = Interpreter(self.dataset)

        # Direct copilot with custom temp storage
        self.copilot = ElephantGoldfishCopilot(self.interp)
        self.copilot.elephant.storage_path = self.memory_file
        self.interp.copilot = self.copilot

    def test_01_goldfish_sliding_window(self):
        """Verify short-term Goldfish buffer records turns and enforces max capacity."""
        buf = GoldfishMemoryBuffer(max_turns=5)
        for i in range(10):
            buf.record_turn('user', f"Message {i}")

        self.assertEqual(len(buf.history), 5)
        self.assertEqual(buf.history[-1]['content'], "Message 9")
        self.assertEqual(buf.history[0]['content'], "Message 5")

    def test_02_elephant_persistent_codebook(self):
        """Verify long-term Elephant memory saves and loads variable codebooks."""
        mem = ElephantLongTermMemory(storage_path=self.memory_file)
        mem.set_codebook_entry("proj1", "mpg", "Miles per gallon fuel economy")
        mem.set_codebook_entry("proj1", "price", "Vehicle price in USD")

        cb = mem.get_codebook("proj1")
        self.assertEqual(cb["mpg"], "Miles per gallon fuel economy")
        self.assertEqual(cb["price"], "Vehicle price in USD")

        # Reload from disk
        mem2 = ElephantLongTermMemory(storage_path=self.memory_file)
        cb2 = mem2.get_codebook("proj1")
        self.assertEqual(cb2["mpg"], "Miles per gallon fuel economy")

    def test_03_contextual_fit_query(self):
        """Fit a model and ask 'what was the R2?' through the Goldfish memory."""
        # Fit regression
        r1 = self.interp.execute("regress price mpg weight")
        self.assertTrue(r1.ok)

        # Ask via AI
        ai_res = self.interp.execute("ai what was the R2?")
        self.assertTrue(ai_res.ok)
        self.assertIn("R-squared", ai_res.output)
        self.assertIn("Model Fit Report", ai_res.output)
        self.assertIn("price", ai_res.output)

    def test_04_contextual_add_option(self):
        """Ask 'add robust standard errors' to modify previous regression in memory."""
        # Fit classical OLS
        self.interp.execute("regress price mpg weight")
        last_before = self.interp.copilot.goldfish.last_command
        self.assertIn("regress price mpg weight", last_before)

        # Contextual follow-up
        ai_res = self.interp.execute("ai add robust standard errors")
        self.assertTrue(ai_res.ok)
        self.assertIn("regress price mpg weight, robust", ai_res.output)
        self.assertIn("Linear regression", ai_res.output)

    def test_05_automated_econometric_interpretation(self):
        """Ask 'explain that' to get natural-language economic significance."""
        self.interp.execute("regress price mpg weight")
        ai_res = self.interp.execute("ai explain that")

        self.assertTrue(ai_res.ok)
        self.assertIn("Interpretation", ai_res.output)
        self.assertIn("Goodness of Fit", ai_res.output)
        self.assertIn("mpg", ai_res.output)
        self.assertIn("weight", ai_res.output)

    def test_06_memory_inspection_and_codebook_ai_commands(self):
        """Test 'ai memory status' and 'ai codebook set' commands."""
        res_mem = self.interp.execute("ai memory status")
        self.assertTrue(res_mem.ok)
        self.assertIn("Memory Status", res_mem.output)
        self.assertIn("Goldfish Buffer", res_mem.output)

        res_cb_set = self.interp.execute("ai codebook set price = Vehicle sticker price")
        self.assertTrue(res_cb_set.ok)
        self.assertIn("Saved codebook entry", res_cb_set.output)

        res_cb_view = self.interp.execute("ai codebook")
        self.assertTrue(res_cb_view.ok)
        self.assertIn("Vehicle sticker price", res_cb_view.output)


if __name__ == '__main__':
    unittest.main()
