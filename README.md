# CLODE (Constrainde Lumping for ODEs)

CLODE is a Python implementation of the algorithm from the paper ''Exact maximal reduction of kinetic models by constrained lumping of differential equations''.

## What is constrained lumping?

Constrained lumping as type of exact order reduction for models defined by a system of ordinary differential equations (ODEs) with polynomial right-hand side.
We will explain it using a toy example. Consider the system

![$\begin{cases} \dot{x}_1  = x_2^2 + 4x_2x_3 + 4x_3^2,\\ \dot{x}_2  =  4x_3 - 2x_1,\\ \dot{x}_3  = x_1 + x_2 \end{cases}$](https://render.githubusercontent.com/render/math?math=%24%5Cbegin%7Bcases%7D%20%5Cdot%7Bx%7D_1%20%20%3D%20x_2%5E2%20%2B%204x_2x_3%20%2B%204x_3%5E2%2C%5C%5C%20%5Cdot%7Bx%7D_2%20%20%3D%20%204x_3%20-%202x_1%2C%5C%5C%20%5Cdot%7Bx%7D_3%20%20%3D%20x_1%20%2B%20x_2%20%5Cend%7Bcases%7D%24)

Assume that we are interested **only** in the dynamics of the variable ![$x_1$](https://render.githubusercontent.com/render/math?math=%24x_1%24). An example of constrained lumping in this case would be the following set of new variables

![$y_1 = x_1 \quad \text{ and } y_2 = x_2 + 2x_3$](https://render.githubusercontent.com/render/math?math=%24y_1%20%3D%20x_1%20%5Cquad%20%5Ctext%7B%20and%20%7D%20y_2%20%3D%20x_2%20%2B%202x_3%24)

The crucial feature of these variables is their derivatives can be written in terms of ![$y_1$](https://render.githubusercontent.com/render/math?math=%24y_1%24) and ![$y_2$](https://render.githubusercontent.com/render/math?math=%24y_2%24) only:

![$\dot{y}_1 = \dot{x}_1 = (x_2 + 2x_3)^2 = y_2^2,$](https://render.githubusercontent.com/render/math?math=%24%5Cdot%7By%7D_1%20%3D%20%5Cdot%7Bx%7D_1%20%3D%20(x_2%20%2B%202x_3)%5E2%20%3D%20y_2%5E2%2C%24)

![$\dot{y}_2 = \dot{x}_2 + 2\dot{x}_3 = 2x_2 + 4x_3 = 2y_2.$](https://render.githubusercontent.com/render/math?math=%24%5Cdot%7By%7D_2%20%3D%20%5Cdot%7Bx%7D_2%20%2B%202%5Cdot%7Bx%7D_3%20%3D%202x_2%20%2B%204x_3%20%3D%202y_2.%24)

Therefore, the original system can be **reduced exactly** to the following system while keeping the variable of interest:

![$\begin{cases} \dot{y}_1 = y_2^2,\\ \dot{y}_2 = 2y_2. \end{cases}$](https://render.githubusercontent.com/render/math?math=%24%5Cbegin%7Bcases%7D%20%5Cdot%7By%7D_1%20%3D%20y_2%5E2%2C%5C%5C%20%5Cdot%7By%7D_2%20%3D%202y_2.%20%5Cend%7Bcases%7D%24)

**In general,** constrained lumping is an exact model reduction by linear transformation that preserves a prescribed set of linear combinations of the unknown functions.
For precise definition and more details, we refer to Section 2 of the paper.
