import jax
from tatbot.utils.log import get_logger

log = get_logger('gpu.wrap')
log.info(f"🧠 JAX devices: {jax.devices()}")