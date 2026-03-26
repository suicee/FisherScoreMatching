"""
pytest configuration: force JAX to use CPU before any tests run.
"""
try:
    import jax
    jax.config.update("jax_platform_name", "cpu")
except ImportError:
    pass
