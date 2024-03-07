#include <iostream>
#include <eigen3/Eigen/Dense>

#include "gradient_descent.h"

using namespace std;
using namespace Eigen;

// Example usage
double sample_function(const VectorXd &x)
{
    // Example function: f(x) = (x +  0.5)^2 + y^2
    // return (x[0] + 0.5) * (x[0] + 0.5) + x[1] * x[1];

    // Task 2: f(x) = [x[0], x[1]] * [[2, -1], [-1], [1]] * [[x[0]], [x[1]]]
    Matrix2d A;
    A << 2, -1,
        -1, 1;
    return x.transpose() * A * x;
}

VectorXd sample_gradient(const VectorXd &x)
{
    // Gradient of the example function: f'(x) = [2x, 2y]
    // return {2 * x[0], 2 * x[1]};

    // Approximate gradient using central differences
    return approximate_gradient(sample_function, x, 1e-6);
}

int main()
{
    size_t count = 10;
    for (size_t i = 0; i < count; i++)
    {
        // Define initial guess and threshold
        Vector2d initial_guess = 10 * Vector2d::Random(2);
        cout << "initial_guess: [" << initial_guess[0] << ", " << initial_guess[1] << "]" << endl;
        double epsilon = 1e-6;

        // Perform gradient descent
        Vector2d result = gradient_descent(sample_function, sample_gradient, initial_guess, epsilon);

        // Print result
        cout << "Minimum point found: [" << result[0] << ", " << result[1] << "]" << endl;
        cout << "F(x) = " << sample_function(result) << endl
             << endl;
    }

    return 0;
}