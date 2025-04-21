import logging
import os
import unittest

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class TestImports(unittest.TestCase):
    def test_ai_module_import(self):
        """Test that ai.py can be imported without errors"""
        try:
            import tatbot.ai
            log.info("Successfully imported ai module")
        except Exception as e:
            self.fail(f"Failed to import ai module: {str(e)}")
            
    def test_evolve_module_import(self):
        """Test that evolve.py can be imported without errors"""
        try:
            import tatbot.ik.evolve
            log.info("Successfully imported evolve module")
        except Exception as e:
            self.fail(f"Failed to import evolve module: {str(e)}")
            
    def test_morph_module_import(self):
        """Test that morph.py can be imported without errors"""
        try:
            import tatbot.ik.morph
            log.info("Successfully imported morph module")
        except Exception as e:
            self.fail(f"Failed to import morph module: {str(e)}")

    def test_mutate_module_import(self):
        """Test that mutate.py can be imported without errors"""
        try:
            import tatbot.ik.mutate
            log.info("Successfully imported mutate module")
        except Exception as e:
            self.fail(f"Failed to import mutate module: {str(e)}")
            

if __name__ == "__main__":
    unittest.main()
