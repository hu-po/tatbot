import time
from tatbot.ik.morph import BaseMorph

class Morph(BaseMorph):
    """
    A template or placeholder morph.

    This morph serves as a basic example and does not perform any inverse kinematics.
    Its primary action in the `_step` method is to pause execution for a specified duration.
    It can be used for testing the simulation pipeline or as a starting point for new morphs.
    """

    def _update_config(self):
        """
        Updates the configuration specific to the Template Morph.

        Sets a short training iteration count and defines the sleep duration
        used in the `_step` method via `config_extras`.
        """
        self.config.train_iters = 2 # Use a minimal number of iterations for this example
        # No step_size needed as no steps are taken
        self.config.config_extras = {
            "sleep_time": 0.01, # Duration to sleep in each step (seconds)
        }

    def _step(self):
        """
        Performs a single step for the Template Morph.

        This implementation simply pauses the execution for the duration specified
        in `self.config.config_extras["sleep_time"]`. It does not modify joint angles.
        """
        time.sleep(self.config.config_extras["sleep_time"])