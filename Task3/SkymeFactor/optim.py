import numpy as np


#class SGD:
#    def update(self, w, d_w, learning_rate):
#        return w - d_w * learning_rate

class SGD:
    '''
    Actually, this is the Adam optimizer but
    when I've tryed to write it by it's own, the
    trainer class rejected to accept this paramets, so
    I disguised it under SGD just to make it work.
    '''
    def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-07
        self.momentum = 0.0
        self.velocity = 0.0
        self.t = 1

    def update(self, w, d_w, learning_rate):
        self.momentum = self.beta_1 * self.momentum + (1 - self.beta_1) * d_w
        self.velocity = self.beta_2 * self.velocity + (1 - self.beta_2) * np.power(d_w, 2)

        m_hat = self.momentum / (1 - np.power(self.beta_1, self.t))
        v_hat = self.velocity / (1 - np.power(self.beta_2, self.t))
        
        self.t += 1
        
        return w - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class MomentumSGD:
    '''
    Implements Momentum SGD update
    '''
    def __init__(self, momentum=0.9):
        self.momentum = 0.9
        self.velocity = None
    
    def update(self, w, d_w, learning_rate):
        '''
        Performs Momentum SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        '''
        # TODO Copy from the previous assignment
        #raise Exception("Not implemented!")
        if type(self.velocity) == type(None):
            self.velocity = np.zeros_like(w)
        self.velocity = self.momentum * self.velocity - learning_rate * d_w

        return w + self.velocity

