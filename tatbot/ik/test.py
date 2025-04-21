import logging
import unittest
import os
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

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

    def test_stencil_module_import(self):
        """Test that stencil.py can be imported without errors"""
        try:
            import tatbot.ik.stencil
            log.info("Successfully imported stencil module")
        except Exception as e:
            self.fail(f"Failed to import stencil module: {str(e)}")
            

if __name__ == "__main__":
    unittest.main()
