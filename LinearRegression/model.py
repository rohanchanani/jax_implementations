from eval import mse, score, multiple_preds, pred
from jax import grad, jit, random, tree_multimap


class LinearRegression:
  
  #Given inputs and correct output, trains a one layer Linear Regression model.
  def train(self, x_data, y_data, num_steps=1000, step_size=0.01, display_info_step=100):
    dimension = x_data.shape[1]
    key = random.PRNGKey(1509)

    #Initialize parameters
    w_key, b_key = random.split(key)
    current_params = {"weights": random.normal(w_key, (dimension,)), "bias": random.normal(b_key)}

    #At each step, updates the parameters with using the gradient of mse function
    def training_step(params, x_multiple, y_multiple, step_size):
      loss_gradients = grad(mse)(params, x_multiple, y_multiple)
      return tree_multimap(lambda param, gradient: param - gradient * step_size, params, loss_gradients)
    
    #Compile training_step function with jit
    jit_training_step = jit(training_step)
    
    #Now the actual training
    for i in range(num_steps):
      current_params = jit_training_step(current_params, x_data, y_data, step_size)
      if display_info_step > 0:
        if i % display_info_step == 0:
          print(f"Step {i} R-Squared: {score(current_params, x_data, y_data)}")

    #Sets the model's coefficients and intercept properties to the final parameters
    self.coefficients = current_params["weights"]
    self.intercept = current_params["bias"]

  #Given an input, returns a prediction using the stored parameters.
  def predict(self, x, multiple=False):
    params = {"weights": self.coefficients, "bias": self.intercept}
    if multiple:
      return multiple_preds(x, params)
    else:
      return pred(x, params)
