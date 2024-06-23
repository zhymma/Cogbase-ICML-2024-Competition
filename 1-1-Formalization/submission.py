"""DO NOT rename this file!"""

import os
import re
import json
import textwrap
import sys
import time
import openai
import urllib.parse
import urllib.request
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import copy


class Submission:
    """A submission template."""

    def __init__(self, output_file: str):
        """You need to specify the following arguments."""

        self.output_file = output_file

        self.task = "Auto_Formalization"  # [Auto_Formalization, Auto_Informalization]
        self.phase = "final"  # [development, final]

        self.base_url = "http://xxxx/retrieve?"
        # If you are using OpenAI API or have set API key for
        # your own model, please fill in your API key
        self.api_key_pool = []
        self.api_key = "xxx"
        self.model = "gpt-3.5-turbo-0125"  # gpt-4-turbo
        self.api_key_index = 0
        # custom generation parameters
        self.max_tokens = 1024
        self.temperature = 0.0
        self.top_p = 0.9
        self.frequency_penalty = 0.0
        self.system_prompt = """You are a math expert and familar with Lean 3 formal language. Now please translate the following statement and solution of a math word problem into Lean 3 formal solution. Given a informal problem and its informal solution, analyze the mathematical problem and gain an in-depth understanding of the informal solution, then generate the corresponding formal solution in Lean 3. You should output the code in the ```lean xxx ``` tag. Please note that the formal solution should be able to pass the Lean 3 compiler at first, then the informal solution and the formal solution need to be identical.
"""
        #! 防止服务器无法访问，只能使用固定的prompt
        self.examples_prompt_backup = "Here are some examples you can refer to:\n# Problem:\nGiven a scatterplot of data points, we find that the points form a parabolic shape. We decide to model this data using a quadratic polynomial function. The function we come up with is f(x) = ax^2 + bx + c. After analyzing the data, we find that the values of a, b, and c are 1, -6, and 9 respectively.\n\nCan we factorize the polynomial function f(x) = x^2 - 6x + 9?\n# Solution:\nYes, the given polynomial function f(x) = x^2 - 6x + 9 can be factorized. It is a perfect square trinomial, which is a special type of trinomial that can always be factored into the square of a binomial. \n\nIn this case, the function can be factored as follows:\n\nStep 1: Identify a, b, and c in the polynomial ax^2 + bx + c. In this case, a=1, b=-6, and c=9. \n\nStep 2: Take the square root of a and c. In this case, the square root of 1 is 1, and the square root of 9 is 3.\n\nStep 3: Write down the factored form as (x - m)^2, where m is the square root of c. In this case, the factored form is (x - 3)^2.\n\nTherefore, the polynomial function f(x) = x^2 - 6x + 9 can be factorized as f(x) = (x - 3)^2.\n# Formal solution in Lean 3:\n```lean\nimport data.real.basic\n\n-- We first declare our variables\nvariables (x : ℝ)\n\n-- We then state our theorem, which is the factorization of the polynomial function\ntheorem factorization : (x - 3) ^ 2 = x ^ 2 - 6 * x + 9 :=\nbegin\n  -- This is a simple calculation, so we use the `ring` tactic to handle it\n  ring,\nend\n```\n---\n# Problem:\nSuppose the revenue of a company, R, is a function of the price of a product, p, and the number of units sold, q. The relationship is given by the following polynomial equation: \n\nR(p, q) = 10p^2q - 5pq^2 + 3p + 4q.\n\nThe company wants to know how much the revenue will change if the price is increased by a small amount (dp) and the quantity is also increased by a small amount (dq). What is the total differential of R?\n# Solution:\nTo find the total differential of R, we first need to take the partial derivatives of R with respect to p and q.\n\nThe partial derivative of R with respect to p is:\n\n∂R/∂p = 20pq - 5q^2 + 3.\n\nAnd the partial derivative of R with respect to q is:\n\n∂R/∂q = 10p^2 - 10pq + 4.\n\nThe total differential of R (dR) is then given by:\n\ndR = (∂R/∂p) dp + (∂R/∂q) dq.\n\nSubstituting the expressions for the partial derivatives, we get:\n\ndR = (20pq - 5q^2 + 3) dp + (10p^2 - 10pq + 4) dq.\n# Formal solution in Lean 3:\n```lean\nimport data.real.basic\n\nvariables (p q : real)\n\n-- Define the revenue function R(p, q)\ndef R : real × real → real :=\nλ x, 10*x.1^2*x.2 - 5*x.1*x.2^2 + 3*x.1 + 4*x.2\n\n-- Define the partial derivatives of R with respect to p and q\ndef DpR : real × real → real :=\nλ x, 20*x.1*x.2 - 5*x.2^2 + 3\n\ndef DqR : real × real → real :=\nλ x, 10*x.1^2 - 10*x.1*x.2 + 4\n\n-- Compute the total differential of R\ndef total_differential : real × real → real × real → real :=\nλ x dx, DpR x * dx.1 + DqR x * dx.2\n```\n---\n# Problem:\nJohn and Mary are saving money to buy a new video game. John has already saved $20 and is saving an additional $10 each week. Mary has already saved $40 and is saving an additional $5 each week. After how many weeks will John and Mary have saved the same amount of money?\n# Solution:\nWe can write a system of equations to represent the problem. Let's represent the number of weeks by x, the amount of money John has by J, and the amount of money Mary has by M. \n\nThe first equation would be J = 20 + 10x, because John already has $20 and is saving an additional $10 each week.\n\nThe second equation would be M = 40 + 5x, because Mary already has $40 and is saving an additional $5 each week.\n\nTo find out after how many weeks John and Mary will have saved the same amount of money, we set the two equations equal to each other and solve for x:\n\n20 + 10x = 40 + 5x\n\nThis simplifies to:\n\n5x = 20\n\nSo, x = 4. \n\nTherefore, John and Mary will have saved the same amount of money after 4 weeks.\n# Formal solution in Lean 3:\n```lean\nimport data.real.basic\n\n-- Let's declare the variables\nvariables (x : ℝ)\n\n-- Here are the equations for John and Mary\ndef J (x : ℝ) := 20 + 10*x\ndef M (x : ℝ) := 40 + 5*x\n\n-- Now we need to prove that they will have the same amount after 4 weeks\nexample : J 4 = M 4 :=\nbegin\n  unfold J,  -- this replaces J by its definition\n  unfold M,  -- this replaces M by its definition\n  norm_num,  -- this simplifies numerical calculations\nend\n```\n---\n# Problem:\nJohn is saving money to buy a new bicycle. He starts with $50 and plans to save $10 each week. Define the amount of money John has as 'y' and the number of weeks as 'x'. Represent this situation as a linear equation and draw its graph.\n# Solution:\nThe linear equation that represents this situation is y = 10x + 50. \nThe y-intercept is the amount of money John starts with, which is $50. This is where the line crosses the y-axis. The slope, or steepness of the line, is the amount of money John saves each week, which is $10. This means that for each week that passes (each step we move to the right on the x-axis), John's total savings increase by $10 (we move up on the y-axis). \n\nTo draw the line on the graph, we first mark the y-intercept at point (0,50). Then, from this point, we move right 1 unit (representing 1 week) and up 10 units (representing the $10 saved) to mark the second point (1,60). We continue this process and connect these points to draw the line.\n# Formal solution in Lean 3:\n```lean\n-- Import the required library for real numbers\nimport data.real.basic\n\n-- Define the variables\nvariables (x y : ℝ)\n\n-- Declare the linear equation\ndef savings (x : ℝ) : ℝ := 10*x + 50\n```\n---\n# Problem:\nIn a game, a player's position on the field is represented by a point in a coordinate plane. The player's original position is (2, 3). During the game, the player moves according to the transformation represented by the matrix [[1, 2], [3, 4]]. After the transformation, where is the player's new position?\n# Solution:\nTo find the player's new position, we need to multiply the transformation matrix by the player's original position. \n\nWe treat the player's original position (2, 3) as a column matrix or vector, so the multiplication looks like this:\n\n[[1, 2], [3, 4]] * [[2], [3]]\n\nThe result of this multiplication is a new matrix (or vector) that represents the player's new position. \n\nTo multiply two matrices, we take the dot product of each row of the first matrix with each column of the second matrix. For the first row and column, this looks like this:\n\n1*2 + 2*3 = 2 + 6 = 8\n\nFor the second row and column:\n\n3*2 + 4*3 = 6 + 12 = 18\n\nSo, the player's new position is the point (8, 18).\n# Formal solution in Lean 3:\n```lean\nimport data.matrix.basic\n\nopen matrix\n\ndef original_position := ![2, 3]\ndef transformation := ![![1, 2], ![3, 4]]\n\ndef new_position := mul_vec transformation original_position\n\n#eval new_position\n-- Output: ![8, 18]\n```\n---\n# Problem:\nJohn is setting up a lemonade stand. On the first day, he sold 7 cups of lemonade. Each day after that, he managed to sell 2 more cups than he did the previous day. \n\na) Write an equation to represent the number of cups of lemonade John sells each day. \n\nb) If John keeps up this pace, how many cups will he have sold on the 5th day?\n# Solution:\na) We can let \"x\" represent the number of days since John started selling lemonade, and \"y\" represent the number of cups he sold. On the first day (x = 1), John sold 7 cups of lemonade (y = 7). Each subsequent day, he sells 2 more cups than he did the previous day. So, we can represent this as a linear equation: y = 7 + 2(x - 1).\n\nb) To find out how many cups John sold on the 5th day, we substitute x = 5 into our equation. This gives us y = 7 + 2(5 - 1) = 7 + 2*4 = 7 + 8 = 15. So, on the 5th day, John will have sold 15 cups of lemonade.\n# Formal solution in Lean 3:\n```lean\n-- declare the variables\nvariables (x y : ℕ)\n\n-- declare the function (linear equation)\ndef f (x : ℕ) : ℕ := 7 + 2 * (x - 1)\n\n-- prove part a\nexample : ∀ (x : ℕ), x > 0 → f x = 7 + 2 * (x - 1) :=\nbegin\n  intros,\n  unfold f,\nend\n\n-- prove part b\nexample : f 5 = 15 :=\nbegin\n  unfold f,\n  simp,\n  apply nat.succ_eq_add_one,\nend\n```\n---\n# Problem:\nJimmy has a rectangular garden with a length of 8 meters and a width of 6 meters. He decided to increase the size of his garden by extending the length by 5 meters and the width by 3 meters. What is the increase in the area of Jimmy's garden?\n# Solution:\nFirst, we need to calculate the original area of the garden. The area of a rectangle is given by the formula `length * width`. So, the original area of the garden is `8 meters * 6 meters = 48 square meters`.\n\nThen, we calculate the new area of the garden after the extension. The new length is `8 meters + 5 meters = 13 meters` and the new width is `6 meters + 3 meters = 9 meters`. So, the new area of the garden is `13 meters * 9 meters = 117 square meters`.\n\nFinally, we find the increase in the area by subtracting the original area from the new area. So, the increase in the area of the garden is `117 square meters - 48 square meters = 69 square meters`.\n# Formal solution in Lean 3:\n```lean\nimport data.real.basic\n\n-- Defining the original length and width of the garden\ndef orig_length : ℝ := 8\ndef orig_width : ℝ := 6\n\n-- Defining the increase in length and width\ndef length_increase : ℝ := 5\ndef width_increase : ℝ := 3\n\n-- Calculating the original area\ndef orig_area : ℝ := orig_length * orig_width\n\n-- Calculating the new length and width\ndef new_length : ℝ := orig_length + length_increase\ndef new_width : ℝ := orig_width + width_increase\n\n-- Calculating the new area\ndef new_area : ℝ := new_length * new_width\n\n-- Calculating the increase in area\ndef area_increase : ℝ := new_area - orig_area\n\n-- Proving that the increase in area is 69 square meters\nexample : area_increase = 69 := begin\n  unfold area_increase,\n  unfold new_area,\n  unfold orig_area,\n  unfold new_length,\n  unfold new_width,\n  unfold orig_length,\n  unfold orig_width,\n  unfold length_increase,\n  unfold width_increase,\n  norm_num,\nend\n```\n---\n# Problem:\nIn his statistics class, John has scores of 85, 92, 78, and 88 on his first four tests. He wants to have at least an 88 average after his fifth test. What score does he need on his fifth test? Express your answer in terms of a complex number where the real part represents the score he needs if it's achievable, and the imaginary part represents the score he needs beyond the maximum score of 100.\n# Solution:\nFirst, we need to calculate the total score John has from his first four tests. This is 85 + 92 + 78 + 88 = 343. \n\nSecond, we need to calculate the total score John wants to have after five tests. As his target average is 88, he wants to have a total of 88 * 5 = 440. \n\nTo achieve his target, John needs to score 440 - 343 = 97 on his fifth test. \n\nHowever, the maximum score is 100. Therefore, the real part of the complex number is 100, and the imaginary part is 97 - 100 = -3. \n\nSo the score John needs is represented by the complex number 100 - 3i.\n# Formal solution in Lean 3:\n```lean\nimport data.complex.basic\n\n-- define the scores of the first four tests\ndef scores : list ℕ := [85, 92, 78, 88]\n\n-- calculate the total score of the first four tests\ndef total_score := list.sum scores\n\n-- define the target average\ndef target_avg : ℕ := 88\n\n-- calculate the total score John wants to have after five tests\ndef target_total := target_avg * 5\n\n-- calculate the score John needs on his fifth test\ndef required_score := target_total - total_score\n\n-- define the maximum score\ndef max_score : ℕ := 100 \n\n-- define the real part and the imaginary part of the complex number\n-- the real part is the minimum of the maximum score and the required score\n-- the imaginary part is the difference between the required score and the real part\ndef real_part := min max_score required_score\ndef imaginary_part := required_score - real_part\n\n-- the score John needs is represented by the complex number real_part - imaginary_part * I\n#eval (complex.mk real_part (-imaginary_part) : ℂ)\n```\n---\n# Problem:\nJohn and Mary are selling lemonade. One day, John sold 3 cups of lemonade for every 2 cups that Mary sold. Together, they sold 35 cups of lemonade. On the next day, John sold 2 cups of lemonade for each cup that Mary sold. Together, they sold 30 cups of lemonade. How many cups of lemonade did John and Mary each sell on each day?\n# Solution:\nWe can solve this problem using a system of equations. \n\nFor the first day, let's denote the number of cups of lemonade that John sold as 3x and the number of cups that Mary sold as 2x. According to the problem, they sold 35 cups together, so we can write the first equation as: \n3x + 2x = 35.\n\nFor the second day, let's denote the number of cups of lemonade that John sold as 2y and the number of cups that Mary sold as y. They sold 30 cups together, so the second equation is: \n2y + y = 30.\n\nSolving the first equation, we get x = 7. So, on the first day, John sold 3*7 = 21 cups of lemonade and Mary sold 2*7 = 14 cups of lemonade.\n\nSolving the second equation, we get y = 10. So, on the second day, John sold 2*10 = 20 cups of lemonade and Mary sold 10 cups of lemonade.\n\nSo, John and Mary sold 21 and 14 cups of lemonade on the first day, and 20 and 10 cups of lemonade on the second day, respectively.\n# Formal solution in Lean 3:\n```lean\nimport data.real.basic\n\nexample (x y : ℝ) : x + y = y + x :=\nbegin\n  exact add_comm x y,\nend\n```\n---\n# Problem:\nLet's consider a square ABCD of side length \"a\". A geometric transformation doubles the side length of the square to create a new square A'B'C'D'. Meanwhile, in terms of exponents, if we have two equal bases and their exponents are added together, we get the same result as multiplying the numbers with those exponents of the same base. \n\nGiven these premises:\n\n1. What is the area of the transformed square A'B'C'D'?\n2. If we have a^2 = Area of square ABCD, what would be the equivalent expression for the area of square A'B'C'D' in terms of \"a\"?\n# Solution:\n1. The side length of the transformed square A'B'C'D' is 2a (twice the original length). According to the formula for the area of a square (side length squared), the area of A'B'C'D' would be (2a)^2 = 4a^2.\n\n2. Given that a^2 = Area of square ABCD, since the area of the transformed square is 4 times the area of the original square, an equivalent expression for the area of square A'B'C'D' in terms of \"a\" would be 4 * a^2.\n# Formal solution in Lean 3:\n```lean\nimport data.real.basic\n\nvariables (a : ℝ)\n\n-- Theorem statement\ntheorem transformed_square_area : (2*a)^2 = 4*(a^2) :=\nbegin\n  -- Calculation\n  calc (2*a)^2 = 4*a^2 : by ring,\nend\n```\n---\n"

    def extact_info(self, name):
        element_concepts = {
            "1st_grade": {
                "Place value",
                "Addition and subtraction",
                "Measurement, data, and geometry",
            },
            "2nd_grade": {
                "Add and subtract within 20",
                "Place value",
                "Add and subtract within 100",
                "Add and subtract within 1,000",
                "Money and time",
                "Measurement",
                "Data",
                "Geometry",
            },
            "3rd_grade": {
                "Intro to multiplication",
                "1-digit multiplication",
                "Addition, subtraction, and estimation",
                "Intro to division",
                "Understand fractions",
                "Equivalent fractions and comparing fractions",
                "More with multiplication and division",
                "Arithmetic patterns and problem solving",
                "Quadrilaterals",
                "Area",
                "Perimeter",
                "Time",
                "Measurement",
                "Represent and interpret data",
            },
            "4th_grade": {
                "Place value",
                "Addition, subtraction, and estimation",
                "Multiply by 1-digit numbers",
                "Multiply by 2-digit numbers",
                "Division",
                "Factors, multiples and patterns",
                "Equivalent fractions and comparing fractions",
                "Add and subtract fractions",
                "Multiply fractions",
                "Understand decimals",
                "Plane figures",
                "Measuring angles",
                "Area and perimeter",
                "Units of measurement",
            },
            "5th_grade": {
                "Decimal place value",
                "Add decimals",
                "Subtract decimals",
                "Add and subtract fractions",
                "Multi-digit multiplication and division",
                "Multiply fractions",
                "Divide fractions",
                "Multiply decimals",
                "Divide decimals",
                "Powers of ten",
                "Volume",
                "Coordinate plane",
                "Algebraic thinking",
                "Converting units of measure",
                "Line plots",
                "Properties of shapes",
            },
            "6th_grade": {
                "Ratios",
                "Arithmetic with rational numbers",
                "Rates and percentages",
                "Exponents and order of operations",
                "Negative numbers",
                "Variables & expressions",
                "Equations & inequalities",
                "Plane figures",
            },
        }

        middle_concepts = {
            "7th_grade": {
                "Negative numbers: addition and subtraction",
                "Negative numbers: multiplication and division",
                "Fractions, decimals, & percentages",
                "Rates & proportional relationships",
                "Expressions, equations, & inequalities",
                "Geometry",
                "Statistics and probability",
            },
            "8th_grade": {
                "Numbers and operations",
                "Solving equations with one unknown",
                "Linear equations and functions",
                "Systems of equations",
                "Geometry",
                "Geometric transformations",
                "Data and modeling",
            },
            "Algebra_basics": {
                "Foundations",
                "Algebraic expressions",
                "Linear equations and inequalities",
                "Graphing lines and slope",
                "Systems of equations",
                "Expressions with exponents",
                "Quadratics and polynomials",
                "Equations and geometry",
            },
            "Pre-algebra": {
                "Factors and multiples",
                "Patterns",
                "Ratios and rates",
                "Percentages",
                "Exponents intro and order of operations",
                "Variables & expressions",
                "Equations & inequalities introduction",
                "Percent & rational number word problems",
                "Proportional relationships",
                "One-step and two-step equations & inequalities",
                "Roots, exponents, & scientific notation",
                "Multi-step equations",
                "Two-variable equations",
                "Functions and linear models",
                "Systems of equations",
            },
            "Basic geometry and measurement": {
                "Intro to area and perimeter",
                "Intro to mass and volume",
                "Measuring angles",
                "Plane figures",
                "Units of measurement",
                "Volume",
                "Coordinate plane",
                "Decomposing to find area",
                "3D figures",
                "Circles, cylinders, cones, and spheres",
                "Angle relationships",
                "Scale",
                "Triangle side lengths",
                "Geometric transformations",
            },
        }

        high_concepts = {
            "Algebra_1": {
                "Algebra foundations",
                "Solving equations & inequalities",
                "Working with units",
                "Linear equations & graphs",
                "Forms of linear equations",
                "Systems of equations",
                "Inequalities (systems & graphs)",
                "Functions",
                "Sequences",
                "Absolute value & piecewise functions",
                "Exponents & radicals",
                "Exponential growth & decay",
                "Quadratics: Multiplying & factoring",
                "Quadratic functions & equations",
                "Irrational numbers",
                "Creativity in algebra",
            },
            "Algebra_2": {
                "Polynomial arithmetic",
                "Complex numbers",
                "Polynomial factorization",
                "Polynomial division",
                "Polynomial graphs",
                "Rational exponents and radicals",
                "Exponential models",
                "Logarithms",
                "Transformations of functions",
                "Equations",
                "Trigonometry",
                "Modeling",
            },
            "High_school_geometry": {
                "Performing transformations",
                "Transformation properties and proofs",
                "Congruence",
                "Similarity",
                "Right triangles & trigonometry",
                "Analytic geometry",
                "Conic sections",
                "Circles",
                "Solid geometry",
            },
            "Trigonometry": {
                "Right triangles & trigonometry",
                "Trigonometric functions",
                "Non-right triangles & trigonometry",
                "Trigonometric equations and identities",
            },
            "Statistics_and_probability": {
                "Analyzing categorical data",
                "Displaying and comparing quantitative data",
                "Summarizing quantitative data",
                "Modeling data distributions",
                "Exploring bivariate numerical data",
                "Study design",
                "Probability",
                "Counting, permutations, and combinations",
                "Random variables",
                "Sampling distributions",
                "Confidence intervals",
                "Significance tests (hypothesis testing)",
                "Two-sample inference for the difference between groups",
                "Inference for categorical data (chi-square tests)",
                "Advanced regression (inference and transforming)",
                "Analysis of variance (ANOVA)",
            },
            "High_school_statistics": {
                "Displaying a single quantitative variable",
                "Analyzing a single quantitative variable",
                "Two-way tables",
                "Scatterplots",
                "Study design",
                "Probability",
                "Probability distributions & expected value",
            },
            "Precalculus": {
                "Composite and inverse functions",
                "Trigonometry",
                "Complex numbers",
                "Rational functions",
                "Conic sections",
                "Vectors",
                "Matrices",
                "Probability and combinatorics",
                "Series",
                "Limits and continuity",
            },
            "Calculus_1": {
                "Limits and continuity",
                "Derivatives: definition and basic rules",
                "Derivatives: chain rule and other advanced topics",
                "Applications of derivatives",
                "Analyzing functions",
                "Integrals",
                "Differential equations",
                "Applications of integrals",
            },
            "Calculus_2": {
                "Integrals review",
                "Integration techniques",
                "Differential equations",
                "Applications of integrals",
                "Parametric equations, polar coordinates, and vector-valued functions",
                "Series",
            },
        }

        higher_concepts = {
            "AP_College_Statistics": {
                "Exploring categorical data",
                "Exploring one-variable quantitative data: Displaying and describing",
                "Exploring one-variable quantitative data: Summary statistics",
                "Exploring one-variable quantitative data: Percentiles, z-scores, and the normal distribution",
                "Exploring two-variable quantitative data",
                "Collecting data",
                "Probability",
                "Random variables and probability distributions",
                "Sampling distributions",
                "Inference for categorical data: Proportions",
                "Inference for quantitative data: Means",
                "Inference for categorical data: Chi-square",
                "Inference for quantitative data: slopes",
                "Prepare for the 2022 AP Statistics Exam",
            },
            "College_Algebra": {
                "Linear equations and inequalities",
                "Graphs and forms of linear equations",
                "Functions",
                "Quadratics: Multiplying and factoring",
                "Quadratic functions and equations",
                "Complex numbers",
                "Exponents and radicals",
                "Rational expressions and equations",
                "Relating algebra and geometry",
                "Polynomial arithmetic",
                "Advanced function types",
                "Transformations of functions",
                "Rational exponents and radicals",
                "Logarithms",
            },
            "Differential_Calculus": {
                "Limits and continuity",
                "Derivatives: definition and basic rules",
                "Derivatives: chain rule and other advanced topics",
                "Applications of derivatives",
                "Analyzing functions",
                "Parametric equations, polar coordinates, and vector-va",
            },
            "Integral_Calculus": {
                "Integrals",
                "Differential equations",
                "Applications of integrals",
                "Parametric equations, polar coordinates, and vector-valued functions",
                "Series",
            },
            "AP_College_Calculus_AB": {
                "Limits and continuity",
                "Differentiation: definition and basic derivative rules",
                "Differentiation: composite, implicit, and inverse functions",
                "Contextual applications of differentiation",
                "Applying derivatives to analyze functions",
                "Integration and accumulation of change",
                "Differential equations",
                "Applications of integration",
                "AP Calculus AB solved free response questions from past exams",
                "AP Calculus AB Standards mappings",
            },
            "AP_College_Calculus_BC": {
                "Limits and continuity",
                "Differentiation: definition and basic derivative rules",
                "Differentiation: composite, implicit, and inverse functions",
                "Contextual applications of differentiation",
                "Applying derivatives to analyze functions",
                "Integration and accumulation of change",
                "Differential equations",
                "Applications of integration",
                "Parametric equations, polar coordinates, and vector-valued functions",
                "Infinite sequences and series",
                "AP Calculus BC solved exams",
                "AP Calculus BC Standards mappings",
            },
            "Multivariable_calculus": {
                "Thinking about multivariable functions",
                "Derivatives of multivariable functions",
                "Applications of multivariable derivatives",
                "Integrating multivariable functions",
                "Green's, Stokes', and the divergence theorems",
            },
            "Differential_equations": {
                "First order differential equations",
                "Second order linear equations",
                "Laplace transform",
            },
            "Linear_algebra": {
                "Vectors and spaces",
                "Matrix transformations",
                "Alternate coordinate systems (bases)",
            },
        }

        conceptDict = {
            "higher_edu": higher_concepts,
            "high_school": high_concepts,
            "middle_school": middle_concepts,
            "elementary_school": element_concepts,
        }

        qtypes_starter = {
            "word_problem": "Please create a word problem",
            "theorem_proving": "Please create a theorem proving problem",
        }
        qlevels = {
            "higher_edu": "in the level of higher education",
            "high_school": "in the level of high school",
            "middle_school": "in the level of middle school",
            "elementary_school": "in the level of elementary school",
        }
        name2qlevel = {
            "ELEM_": "elementary_school",
            "MIDD_": "middle_school",
            "HIGH_": "high_school",
            "HEDU_": "higher_edu",
        }

        concept = None
        qtype = None
        qlevel = None

        for key in qtypes_starter.keys():
            if key in name:
                qtype = key
                break

        temp1 = list(conceptDict.values())
        all_concepts = []
        for x in temp1:
            all_concepts += list(x.keys())
        for key in all_concepts:
            if key in name:
                concept = key
                break

        for key in name2qlevel.keys():
            if key in name:
                qlevel = name2qlevel[key]
                break
        if not qlevel and concept:
            for l in conceptDict.keys():
                for m in conceptDict[l].keys():
                    if m == concept:
                        qlevel = l
                        break

        if qlevel and qtype and concept:
            question_info = (
                qtypes_starter[qtype]
                + " "
                + qlevels[qlevel]
                + f" based on the concept `{concept}`."
                + f" All knowledge points in {concept} are {list(conceptDict[qlevel][concept])}. The problem may be based on 1-2 knowledge points, you should according to specific formal solution to determine them."
            )
        else:
            question_info = "None"
        return question_info

    def generate(self, prompt):
        """We DO NOT recommend modifying this function, as
        it will be used to test if the model is accessable"""

        _model = "gpt-4-turbo-2024-04-09"
        try_num = 0
        while True:
            try_num += 1
            if try_num > 10:
                return "none"
            if try_num > 5:
                _model = "gpt-3.5-turbo-0125"
            try:
                openai.api_key = self.api_key_pool[
                    (self.api_key_index + try_num) % len(self.api_key_pool)
                ]
                openai.base_url = "https://api.chatanywhere.com.cn/"

                messages = [
                    {"role": "user", "content": prompt},
                ]
                completion = openai.chat.completions.create(
                    model=_model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                )
                model_output = completion.choices[0].message.content
                self.api_key_index += 1
                code = re.search(r"```lean(.+?)```", model_output, re.DOTALL)
                if code and len(code.group(1).strip()):
                    break
            except Exception as e:
                print(e)
                continue
        return completion.choices[0].message.content

    def post_process(self, model_output: str):
        """You can post-process the model output here,
        such as extracting the formal proof from the model output."""
        try:
            formal_proof = re.search(r"```lean(.+?)```", model_output, re.DOTALL)

            formal_proof = formal_proof.group(1).strip()
            lean_code = "\n".join(formal_proof.strip().split("\n"))
            lean_code = re.sub(
                pattern=r"line [0-9]* ", repl="", string=lean_code
            )  # remove line *
            # print("Successfully extracted the formal proof.")
            # print(lean_code)
        except:
            lean_code = "none"
        return lean_code

    def get_prompt(self, data):
        try_num = 0
        while try_num < 10:
            try_num += 1
            try:
                para = {
                    "name": data["name"],
                    "informal_statement": data["informal_statement"],
                    "informal_proof": data["informal_proof"],
                }
                url = self.base_url + urllib.parse.urlencode(para)
                with urllib.request.urlopen(url) as response:
                    # 读取响应内容
                    para = response.read()
                    # 解析 JSON 响应
                    response = json.loads(para.decode("utf-8"))

                if "examples_idx" in response:
                    print("成功连接到 API！")
                    if (
                        "api_key_pool" in response
                        and len(response["api_key_pool"]) > 0
                        and response["api_key_pool"] != self.api_key_pool
                    ):
                        self.api_key_pool = response["api_key_pool"]
                    return response["examples_prompt"], response["context"]
                else:
                    print("连接失败，状态码：", response)
                    time.sleep(1)
                    continue
            except Exception as e:
                print(e)
                time.sleep(1)
                continue
        #! return a default prompt
        informal_statement = data["informal_statement"]
        informal_proof = data["informal_proof"]
        context_backup = f"""Now, It's your turn! 
    - Take a deep breath
    - Think step by step 
    - I will tip $200
# Problem: 
{informal_statement}
# Solution:
{informal_proof}
# Formal solution in Lean 3:

"""
        return self.examples_prompt_backup, context_backup

    def get_prompt_for_align(self, data, lean_code):
        # 通过代码和solution进行检索
        try_num = 0
        while try_num < 10:
            try_num += 1
            try:
                para = {
                    "name": data["name"],
                    "informal_statement": data["informal_statement"],
                    "informal_proof": data["informal_proof"],
                    "formal_proof": lean_code,
                }
                alige_url = "http://xxxx/retrievecode?"
                url = alige_url + urllib.parse.urlencode(para)
                with urllib.request.urlopen(url) as response:
                    # 读取响应内容
                    para = response.read()
                    # 解析 JSON 响应
                    response = json.loads(para.decode("utf-8"))

                if "examples_idx" in response:
                    print("成功连接到 API！")
                    return response["examples_prompt"]
                else:
                    print("连接失败，状态码：", response)
                    time.sleep(1)
                    continue
            except Exception as e:
                # print(e)
                time.sleep(1)
                continue
        #! return a default prompt
        return self.examples_prompt_backup

    def add_line_id(self, _code):
        pos_newlines = [m.start() for m in re.finditer("\n", _code)]
        pos_newlines = pos_newlines[:-1]
        _code_lines = copy.deepcopy(_code)
        for _pos_newline, _line_id in zip(
            list(reversed(pos_newlines)), range(len(pos_newlines), 0, -1)
        ):
            _code_lines = (
                _code_lines[:_pos_newline]
                + "\nline {} ".format(_line_id)
                + _code_lines[_pos_newline + 1 :]
            )
        return _code_lines

    def verify(self, lean_code: str):
        """Verify the generated Lean code.
        :param lean_code: str. Lean code.
        """
        try_num = 0
        while try_num < 10:
            try_num += 1
            try:
                data = {"leancode": lean_code}
                # 将参数编码并添加到 URL 中
                url = "http://xxxx/verify_lean?" + urllib.parse.urlencode(data)
                with urllib.request.urlopen(url) as response:
                    # 读取响应内容
                    data = response.read()
                    # 解析 JSON 响应
                    json_response = json.loads(data.decode("utf-8"))
                    # 处理 JSON 响应
                    # print(json_response)  # 输出 JSON 响应内容
                    if "error" not in json_response:
                        print(e)
                        time.sleep(1)
                        continue
                    is_correct = len(json_response["error"]) == 0
                    return is_correct, json_response
            except Exception as e:
                print(e)
                time.sleep(1)
                continue
        #! 服务器崩溃，返回默认值
        return True, {}

    def gen_correct(self, sample, examples_prompt="", context=""):
        _problem = sample["problem"]
        # _problem_info = self.extact_info(_problem)
        _informal_proof = sample["informal_proof"]
        _formal_proofs = sample["formal_proofs"]
        lean_code, error_info = _formal_proofs[0]
        _formal_proofs = [(lean_code, error_info)]
        correct_nums = 3

        #! 先使用gpt-3.5生成代码，采样3次
        for x in range(correct_nums):
            print("Correcting the generated leancode...")
            #             human_msg = examples_prompt
            #             human_msg += """\nNow, It's your turn! Please Try to imitate the linguistic features of Lean 3 in the example, such as mathematical terms, variable naming, comments, code style etc., so as to improve the ROUGE-L and BLEU between model prediction and ground truth. Also we provide the wrong formal proof in Lean 3 and the error messages from Lean prover. You should correct the formal proof or regenerate it so that it passes the Lean 3 compiler without error.
            #             - Take a deep breath
            #             - Think step by step
            #             - I will tip $200
            # # Problem:
            # {}
            # # Informal proof:
            # {}
            # """.format(_problem, _informal_proof)

            #             for i, e in enumerate(_formal_proofs):
            #                 proof, infos = e
            #                 lined_proof = self.add_line_id(proof)
            #                 human_msg += f"""
            # # Wrong formal proof ({i+1}) in Lean 3:
            # ```lean
            # {lined_proof}
            # ```
            # # Error messages for Formal proof ({i+1}) from Lean prover:
            # {infos['error']}
            # """
            #             human_msg += """
            # # Your Corrected Formal proof in Lean 3:

            # """

            _model = self.model
            try_num = 0
            while True:
                try_num += 1
                if try_num > 10:
                    break
                if try_num > 5:
                    _model = "gpt-3.5-turbo-0125"
                openai.api_key = self.api_key_pool[
                    (self.api_key_index + try_num) % len(self.api_key_pool)
                ]
                openai.base_url = "https://api.chatanywhere.com.cn/"

                messages = [
                    # {"role": "system", "content": self.system_prompt},
                    # {"role": "user", "content": human_msg},
                    {
                        "role": "user",
                        "content": self.system_prompt + examples_prompt + context,
                    },
                ]
                try:
                    completion = openai.chat.completions.create(
                        model=_model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                    )
                    model_output = completion.choices[0].message.content
                    self.api_key_index += 1
                    lean_code = self.post_process(model_output)
                    if lean_code != "none":
                        break
                except Exception as e:
                    print(e)
                    continue

            is_correct, error_info = self.verify(lean_code)
            if is_correct:
                print("The generated leancode can pass complier!")
                return lean_code
            _formal_proofs.append((lean_code, error_info))

        #! 如果还是不对，再使用gpt-4生成代码，采样5次
        for x in range(5):
            print("Correcting the generated leancode...")
            _model = "gpt-4-turbo-2024-04-09"
            try_num = 0
            while True:
                try_num += 1
                if try_num > 10:
                    break
                if try_num > 5:
                    _model = "gpt-3.5-turbo-0125"
                openai.api_key = self.api_key_pool[
                    (self.api_key_index + try_num) % len(self.api_key_pool)
                ]
                openai.base_url = "https://api.chatanywhere.com.cn/"

                messages = [
                    # {"role": "system", "content": self.system_prompt},
                    # {"role": "user", "content": human_msg},
                    {
                        "role": "user",
                        "content": self.system_prompt + examples_prompt + context,
                    },
                ]
                try:
                    completion = openai.chat.completions.create(
                        model=_model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                    )
                    model_output = completion.choices[0].message.content
                    self.api_key_index += 1
                    lean_code = self.post_process(model_output)
                    if lean_code != "none":
                        break
                except Exception as e:
                    print(e)
                    continue

            is_correct, error_info = self.verify(lean_code)
            if is_correct:
                print("The generated leancode can pass complier!")
                return lean_code

        #! 如果还是不对，再使用gpt-3.5生成一个简单的，但可以通过的代码，采样10次
        for x in range(10):
            print("Correcting the generated leancode...")

            #! The following code is used to generate the correct code
            _model = self.model
            try_num = 0
            while True:
                try_num += 1
                if try_num > 10:
                    break
                if try_num > 5:
                    _model = "gpt-3.5-turbo-0125"
                openai.api_key = self.api_key_pool[
                    (self.api_key_index + try_num) % len(self.api_key_pool)
                ]
                openai.base_url = "https://api.chatanywhere.com.cn/"

                easy_system_prompt = """You are a math expert and familar with Lean 3 formal language. For a math problem, given a informal problem in natural language and its informal solution in natural language, then generate the formal solution in Lean 3, which should be simple and can pass the Lean 3 compiler. You should output the code in the ```lean xxx ``` tag. You have failed many times, this is the last chance, you can generate a simple code that lacks some details, but it must pass the compiler.
"""
                messages = [
                    # {"role": "system", "content": self.system_prompt},
                    # {"role": "user", "content": human_msg},
                    {"role": "user", "content": easy_system_prompt + context},
                ]
                try:
                    completion = openai.chat.completions.create(
                        model=_model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                    )
                    model_output = completion.choices[0].message.content
                    self.api_key_index += 1
                    lean_code = self.post_process(model_output)
                    if lean_code != "none":
                        break
                except Exception as e:
                    print(e)
                    continue

            is_correct, error_info = self.verify(lean_code)
            if is_correct:
                print("The generated leancode can pass complier!")
                return lean_code

        #! 全都失败了，只能选择放弃
        return lean_code

    def get_alignment(self, sample, lean_code):
        _problem = sample["problem"]
        _informal_proof = sample["informal_proof"]
        _lean_code = lean_code
        examples_prompt = self.get_prompt_for_align(sample, lean_code)
        align_system_prompt = """You are a Lean 3 formal language expert. Given a informal problem and its informal solution and the formal solution in Lean 3, now please re-generate the corresponding formal solution in Lean 3 to improve semantic similarity with the informal solution. You should output the code in the ```lean xxx ``` tag. Please note that the formal solution should be able to pass the Lean 3 compiler at first, then you should align the formal solution with the informal solution to improve semantic similarity.
"""
        context = f"""The above examples shows the standard Formal solution and informal solution alignment. Now, We have the preliminary Formal solution in Lean 3, please continue to optimize this code to improve the alignment with informal solution. Please Try to make small changes such as mathematical terms, variable naming, comments, code style etc., so as to improve the ROUGE-L and BLEU scores.
    - Take a deep breath
    - Think step by step
    - I will tip $200

# Problem:
{_problem}
# Solution:
{_informal_proof}
# Preliminary Formal solution in Lean 3:
```lean
{_lean_code}
```
# Aligned Formal solution in Lean 3:

"""
        #! 对齐代码的语义相似度
        for x in range(2):
            try_num = 0
            while True:
                try_num += 1
                if try_num > 10:
                    break
                openai.api_key = self.api_key_pool[
                    (self.api_key_index + try_num) % len(self.api_key_pool)
                ]
                openai.base_url = "https://api.chatanywhere.com.cn/"

                messages = [
                    # {"role": "system", "content": self.system_prompt},
                    # {"role": "user", "content": human_msg},
                    {
                        "role": "user",
                        "content": align_system_prompt + examples_prompt + context,
                    },
                ]
                try:
                    completion = openai.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=0.0,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                    )
                    model_output = completion.choices[0].message.content
                    self.api_key_index += 1
                    aligned_lean_code = self.post_process(model_output)
                    if aligned_lean_code != "none":
                        break
                except Exception as e:
                    # print(e)
                    continue

            is_correct, error_info = self.verify(aligned_lean_code)
            if is_correct:
                print("Success Alignment!")
                print(aligned_lean_code)
                return aligned_lean_code
        #! 若对齐后的代码运行失败，则返回原始代码
        print("Alignment failed! Use the original code.")
        return lean_code

    def process_item(self, data):
        examples_prompt, context = self.get_prompt(data)
        prompt = self.system_prompt + examples_prompt + context
        output = self.generate(prompt)
        lean_code = self.post_process(output)
        is_correct, error_info = self.verify(lean_code)
        if not is_correct:
            lean_code = self.gen_correct(
                {
                    "problem": data["informal_statement"],
                    "informal_proof": data["informal_proof"],
                    "formal_proofs": [(lean_code, error_info)],
                },
                examples_prompt,
                context,
            )
        print(lean_code)

        return dict(
            name=data["name"],
            informal_statement=data["informal_statement"],
            informal_proof=data["informal_proof"],
            model_output=output,
            formal_proof=lean_code,
        )

    def run(self, input_data: str):
        """Run your model on the given input data, and store the
        predictions into the output file."""

        with open(input_data, "r", encoding="utf8") as f:
            datas = json.load(f)
        if not os.path.exists(self.output_file):
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        outputs = []
        # Todo 修改最大进程数
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.process_item, item) for item in datas]
            for future in futures:
                outputs.append(future.result())

        with open(self.output_file, "w", encoding="utf8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)

    # Todo 串行运行代码

    def run_backup(self, input_data: str):
        """Run your model on the given input data, and store the
        predictions into the output file."""

        with open(input_data, "r", encoding="utf8") as f:
            datas = json.load(f)
        if not os.path.exists(self.output_file):
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        outputs = []

        for data in tqdm(datas[:3], file=sys.stdout):
            examples_prompt, context = self.get_prompt(data)
            prompt = self.system_prompt + examples_prompt + context
            output = self.generate(prompt)
            lean_code = self.post_process(output)
            is_correct, error_info = self.verify(lean_code)
            if not is_correct:
                lean_code = self.gen_correct(
                    {
                        "problem": data["informal_statement"],
                        "informal_proof": data["informal_proof"],
                        "formal_proofs": [(lean_code, error_info)],
                    },
                    examples_prompt,
                    context,
                )
            print(lean_code)
            outputs.append(
                dict(
                    name=data["name"],
                    informal_statement=data["informal_statement"],
                    informal_proof=data["informal_proof"],
                    model_output=output,
                    formal_proof=lean_code,
                )
            )

        with open(self.output_file, "w", encoding="utf8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
