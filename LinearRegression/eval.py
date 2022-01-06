from jax import numpy as jnp, vmap

def pred(x, params):
  return jnp.dot(params["weights"], x) + params["bias"]

multiple_preds = vmap(pred, (0, None))

def mse(params, x_multiple, y_multiple):
  print(x_multiple.shape)
  prediction = multiple_preds(x_multiple, params)
  actual = y_multiple
  return jnp.mean(jnp.multiply(prediction - actual, prediction - actual))

def score(params, x_multiple, y_multiple):
  prediction = multiple_preds(x_multiple, params)
  actual = y_multiple
  return 1 - (jnp.dot(prediction - actual, prediction - actual) / jnp.dot(actual - jnp.mean(actual), actual - jnp.mean(actual)))