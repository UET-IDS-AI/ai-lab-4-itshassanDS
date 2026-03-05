[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/9TnUh6ww)
# AI Lab — Gradient Descent + Linear Regression

This GitHub Classroom assignment has **4 questions**:

## Q1 — Implement & visualize gradient descent
Implement gradient descent for linear regression and produce the arrays needed to visualize:
- Loss vs epoch
- Parameter path `(theta0, theta1)` in parameter space

Reference (inspiration): AML Book Lecture 3 (Linear Regression)  
https://kuleshov-group.github.io/aml-book/contents/lecture3-linear-regression.html

**You will implement:**
- `gradient_descent_linreg(...)`
- `visualize_gradient_descent(...)`

---

## Q2 — Diabetes linear regression using gradient descent
Fit linear regression on the **sklearn diabetes dataset** using your GD implementation.

**You will implement:**
- `diabetes_linear_gd(...)`

Return: `(train_mse, test_mse, train_r2, test_r2, theta)`

---

## Q3 — Diabetes linear regression using analytical solution
Implement the closed-form solution (normal equation) with a tiny ridge term for numerical stability:

\[
\theta = (X^T X + \lambda I)^{-1} X^T y
\]

**You will implement:**
- `diabetes_linear_analytical(...)`

Return: `(train_mse, test_mse, train_r2, test_r2, theta)`

---

## Q4 — Compare GD vs Analytical solution
Fit both methods and compare:
- coefficient L2 difference
- cosine similarity
- differences in train/test metrics

**You will implement:**
- `diabetes_compare_gd_vs_analytical(...)`

Return a dictionary with:
- `theta_l2_diff`
- `train_mse_diff`
- `test_mse_diff`
- `train_r2_diff`
- `test_r2_diff`
- `theta_cosine_sim`
