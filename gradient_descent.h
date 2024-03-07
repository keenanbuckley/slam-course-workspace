#ifndef _GRADIENT_DESCENT_H
#define _GRADIENT_DESCENT_H

#include <iostream>
#include <cmath>
#include <functional>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

// Function to compute the L2 norm of a vector
double l2_norm(const VectorXd &v);

// Gradient descent function with backtracking line search
VectorXd gradient_descent(const function<double(const VectorXd &)> &f,
                          const function<VectorXd(const VectorXd &)> &grad_f,
                          const VectorXd &initial_x,
                          double epsilon = 1e-4,
                          double alpha = 1.0,
                          double beta = 0.5,
                          int max_iter = 1000);

// Compute approximate gradient around a point using central differences
VectorXd approximate_gradient(  const function<double(const VectorXd &)> &f,
                                const VectorXd &x,
                                double epsilon = 1e-6);

#endif