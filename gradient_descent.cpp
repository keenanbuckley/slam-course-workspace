#include "gradient_descent.h"

// Function to compute the L2 norm of a vector
double l2_norm(const VectorXd &v)
{
    double norm = 0.0;
    for (double val : v)
    {
        norm += val * val;
    }
    return sqrt(norm);
}

// Gradient descent function with backtracking line search
VectorXd gradient_descent(const function<double(const VectorXd &)> &f,
                          const function<VectorXd(const VectorXd &)> &grad_f,
                          const VectorXd &initial_x,
                          double epsilon,
                          double alpha,
                          double beta,
                          int max_iter)
{
    VectorXd x(initial_x.rows());
    x = initial_x;
    int iter = 0;
    while (iter < max_iter)
    {
        VectorXd grad = grad_f(x);
        double grad_norm = l2_norm(grad);
        if (grad_norm < epsilon)
        {
            cout << "Gradient descent converged with L2 norm of gradient below threshold." << endl;
            break;
        }

        VectorXd new_x(x.rows());
        double t = 1.0;
        do
        {
            new_x = x - t * alpha * grad_f(x);
            t *= beta;
        } while (f(new_x) >= f(x));

        x = new_x;
        iter++;
    }

    if (iter >= max_iter)
    {
        cout << "Maximum number of iterations reached without convergence." << endl;
    }

    return x;
}

// Compute approximate gradient around a point using central differences
VectorXd approximate_gradient(const function<double(const VectorXd &)> &f,
                              const VectorXd &x,
                              double epsilon)
{
    int n = x.size();
    VectorXd grad(n);
    for (int i = 0; i < n; ++i)
    {
        VectorXd x_plus = x;
        x_plus(i) += epsilon;
        VectorXd x_minus = x;
        x_minus(i) -= epsilon;
        grad(i) = (f(x_plus) - f(x_minus)) / (2 * epsilon);
    }
    return grad;
}