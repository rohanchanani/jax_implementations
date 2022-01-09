from jax import numpy as jnp, vmap

<<<<<<< HEAD
#Returns a prediction value given inputs and parameters
def pred(x, params):
  return jnp.dot(params["weights"], x) + params["bias"]

#Vectorized version of prediction file for batches of inputs (multiple rows)
multiple_preds = vmap(pred, (0, None))

#Given parameters, batch of inputs, and batch of corresponding true outputs, returns mean squared error.
=======
def pred(x, params):
  return jnp.dot(params["weights"], x) + params["bias"]

multiple_preds = vmap(pred, (0, None))

>>>>>>> 3f0ff231053bccdc029852161231df96bb2ccf74
def mse(params, x_multiple, y_multiple):
  print(x_multiple.shape)
  prediction = multiple_preds(x_multiple, params)
  actual = y_multiple
  return jnp.mean(jnp.multiply(prediction - actual, prediction - actual))

<<<<<<< HEAD
#Given parameters, batch of inputs, and batch of corresponding true outputs, returns R^2 value.
=======
>>>>>>> 3f0ff231053bccdc029852161231df96bb2ccf74
def score(params, x_multiple, y_multiple):
  prediction = multiple_preds(x_multiple, params)
  actual = y_multiple
  return 1 - (jnp.dot(prediction - actual, prediction - actual) / jnp.dot(actual - jnp.mean(actual), actual - jnp.mean(actual)))