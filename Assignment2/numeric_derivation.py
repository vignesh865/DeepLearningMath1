
def derive(f, x, h=0.0001):

    x_result = f(x)
    x_h_result = f(x+h)

    return (x_h_result - x_result) / h   # TODO: implement this function
