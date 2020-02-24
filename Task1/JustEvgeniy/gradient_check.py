import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    print('check_gradient'.center(40))
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()
    
    print(' - analytic_grad'.ljust(40), '=', analytic_grad)

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        print(' - ix'.ljust(40), '=', ix)
        
        single_x = x[ix[0]]
        print(' - single_x'.ljust(40), '=', single_x)
        
        x_plus = x.copy()
        x_plus[ix] += delta
        x_minus = x.copy()
        x_minus[ix] -= delta
        print(' - x_plus'.ljust(40), '=', x_plus)
        print(' - x_minus'.ljust(40), '=', x_minus)

        f_plus_delta = f(x_plus)
        f_minus_delta = f(x_minus)
        print(' - f(x_plus)'.ljust(40), '=', f_plus_delta)
        print(' - f(x_minus)'.ljust(40), '=', f_minus_delta)
     
        val_at_plus = f_plus_delta[0]
        val_at_minus = f_minus_delta[0]
        print(' - val_at_plus'.ljust(40), '=', val_at_minus)
        print(' - val_at_minus'.ljust(40), '=', val_at_minus)
        
        numeric_grad_at_ix = (val_at_plus - val_at_minus) / (2 * delta)
        print(' - numeric_grad_at_ix'.ljust(40), '=', numeric_grad_at_ix)
        
        analytic_grad_at_ix = analytic_grad[ix]
        print(' - analytic_grad_at_ix'.ljust(40), '=', analytic_grad_at_ix)
        
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            print('END check_gradient END'.center(40))
            return False

        it.iternext()

    print("Gradient check passed!")
    print('END check_gradient END'.center(40))
    return True
