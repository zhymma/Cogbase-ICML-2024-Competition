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

        self.task = "Auto_Informalization"  # [Auto_Formalization, Auto_Informalization]
        self.phase = "final"  # [development, final]

        self.base_url = "http://xxxx/2retrieve?"
        # If you are using OpenAI API or have set API key for
        # your own model, please fill in your API key
        self.api_key_pool = []
        self.api_key = "xxx"
        self.model = "gpt-4"  # gpt-3.5-turbo-0125 , gpt-4
        self.api_key_index = 0
        # custom generation parameters
        self.max_tokens = 1024
        self.temperature = 0.7
        self.top_p = 1.0
        self.frequency_penalty = 0.0
        self.system_prompt = """You are a math expert. Now please come up with a math problem according to the following requirements. The math problem should contain a question part (indicated by ``# Problem: ''), a corresponding informal solution in natural language (indicated by ``# Solution: ''), and a corresponding formal solution in Lean 3 (indicated by ``# Formal solution in Lean 3: '').  Now we provide the formal solution in Lean 3, please generate the corresponding informal solution first, then generate the problem. Please note that the informal solution and the formal solution need to be identical.
Also, here are some instruction you should follow. 
1. Consistency in Mathematical Formalization:
   - Ensure consistency between formal proofs and problem statements in terms of mathematical concepts and notation to enhance relevance.
2. generated questionIt should be a word problem with a backstory of a real-life scenario, not an abstract math problem.

3. Ensure the fluency and naturalness of the language of mathematical problems and solutions.

"""
        self.example_prompt_backup = """Here are some examples you can refer to:
# Formal solution in Lean 3:
```lean
import data.real.basic\n\n-- Definitions\nnoncomputable def center_of_rectangle (x1 x2 y1 y2 : ℝ) := ((x1+x2)/2, (y1+y2)/2)\ndef area_of_rectangle (length width : ℝ) := length * width\n\n-- Proofs\nexample : center_of_rectangle 0 12 0 8 = (6, 4) :=\nbegin\n  unfold center_of_rectangle,\n  norm_num,\nend\n\nexample : area_of_rectangle 12 8 = 96 :=\nbegin\n  unfold area_of_rectangle,\n  norm_num,\nend
```
# Problem
Jenny is planning to make a rectangular garden plot in her yard. She wants the length of the plot to be 12 feet and the width to be 8 feet. She marks the four corners of the plot on a coordinate plane as follows: (0,0), (12,0), (0,8), and (12,8). \n\n1. What are the coordinates of the center of the plot? \n2. How many square feet is the garden plot?

# Solution
1. The center of the plot is the average of the x-coordinates and the y-coordinates. So, the x-coordinate of the center is (0+12)/2 = 6 and the y-coordinate of the center is (0+8)/2 = 4. Therefore, the center of the plot is at (6,4).\n\n2. The area of a rectangle is calculated by multiplying the length by the width. Here, the length is 12 feet and the width is 8 feet, so the area is 12*8 = 96 square feet. Therefore, the garden plot is 96 square feet.

---

# Formal solution in Lean 3:
```lean
import data.real.basic\nimport data.nat.basic\n\n-- define the quadratic equation\ndef quadratic_eq (x : ℝ) : Prop := x^2 - 4*x + 4 = 0\n\n-- proof of the solutions to the quadratic equation\nlemma solve_quadratic_eq : ∃ x : ℝ, quadratic_eq x :=\nbegin\n  use 2,\n  unfold quadratic_eq,\n  norm_num,\nend\n\n-- define the chi-square test\ndef chi_square_test (df : ℕ) (test_stat : ℝ) : Prop := df = 3 ∧ test_stat = 7.815\n\n-- proof of the result of the chi-square test\nlemma result_chi_square_test : ∀ (df : ℕ) (test_stat : ℝ), chi_square_test df test_stat → ¬ (test_stat > 7.815) :=\nbegin\n  intros df test_stat h,\n  cases h with h1 h2,\n  linarith,\nend
```

# Problem
Jenny is planning to make a rectangular garden plot in her yard. She wants the length of the plot to be 12 feet and the width to be 8 feet. She marks the four corners of the plot on a coordinate plane as follows: (0,0), (12,0), (0,8), and (12,8). \n\n1. What are the coordinates of the center of the plot? \n2. How many square feet is the garden plot?

# Solution
1. The center of the plot is the average of the x-coordinates and the y-coordinates. So, the x-coordinate of the center is (0+12)/2 = 6 and the y-coordinate of the center is (0+8)/2 = 4. Therefore, the center of the plot is at (6,4).\n\n2. The area of a rectangle is calculated by multiplying the length by the width. Here, the length is 12 feet and the width is 8 feet, so the area is 12*8 = 96 square feet. Therefore, the garden plot is 96 square feet.

---

# Formal solution in Lean 3:
```lean
import data.real.basic\n\n-- defining area of a square\ndef square_area (a : ℝ) : ℝ := a * a\n\n-- defining perimeter of a square\ndef square_perimeter (a : ℝ) : ℝ := 4 * a\n\n-- proving that area of a square with side 5 is 25\nexample : square_area 5 = 25 := \nbegin \n    unfold square_area, \n    norm_num, \nend\n\n-- proving that perimeter of a square with side 5 is 20\nexample : square_perimeter 5 = 20 := \nbegin \n    unfold square_perimeter, \n    norm_num, \nend
```

# Problem
Given a square with a side length of 5 units. What is the area of the square and the perimeter of the square?

# Solution
The area of a square is calculated by squaring the length of one side. So, the area of a square with each side of 5 units would be 5*5=25 square units.\n\nThe perimeter of a square is calculated by multiplying the length of one side by 4. So, the perimeter of a square with each side of 5 units would be 5*4=20 units.

---
"""
        self.prompt_retrieve = """
You are a math expert and familiar with Lean 3.
Given {num} math questions along with their solutions and a Lean 3 code,
You should identify which question along with solution best match the provided Lean 3 code.  In other words, I'll determine which question the code aims to solve and confirm if the code is a Lean 3 implementation of the solution.
Rank the {num} questions above based on their relevance to the Lean 3 code. The questions should be listed in descending order using identifers, and the most relevant questions should be listed first, and the output format of ranking results should be a list of identifer numbers. Only response the ranking results, do not say any word or explain.
Here are some instructions to follow:
- Output the ranking results based on the relevance of the questions to the code.
- Ensure that the most relevant question is ranked first in the list.
For Example:
[1,2,0]

- Take a deep breath
- Think step by step 
- I will tip $200

# Lean 3 Code: 
```lean
{code}
```

# Questions with Solutions:
{questions}

# Ranking Results:

"""

    def generate(self, prompt):
        """We DO NOT recommend modifying this function, as
        it will be used to test if the model is accessable"""
        _model = self.model
        try_num = 0
        while True:
            try_num += 1
            if try_num > 5:
                _model = "gpt-3.5-turbo-0125"
            if try_num > 10:
                return ""
            openai.api_key = self.api_key_pool[
                (self.api_key_index + try_num) % len(self.api_key_pool)
            ]
            openai.base_url = "https://api.chatanywhere.com.cn/"

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
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
                solution = re.search(
                    r"# Solution:(.+?)# Problem:", model_output, re.DOTALL
                )
                problem = re.search(r"# Problem:(.+)", model_output, re.DOTALL)
                if problem and solution:
                    break
            except Exception as e:
                print("-" * 20)
                print(e)
                print("API key error, try again")
                # print(model_output)
                print("-" * 20)
                time.sleep(60)
                continue
        return completion.choices[0].message.content

    def get_gpt_output(self, prompt, model, temperature):
        """We DO NOT recommend modifying this function, as
        it will be used to test if the model is accessable"""
        ranking_list = [0]
        try_num = 0
        _model = model
        while True:
            try_num += 1
            if try_num > 5:
                _model = "gpt-3.5-turbo-0125"
            if try_num > 10:
                return ranking_list
            openai.api_key = self.api_key_pool[
                (self.api_key_index + try_num) % len(self.api_key_pool)
            ]
            openai.base_url = "https://api.chatanywhere.com.cn/"

            messages = [
                {"role": "system", "content": "You an expert on code and retrieve"},
                {"role": "user", "content": prompt},
            ]
            try:

                completion = openai.chat.completions.create(
                    model=_model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                )
                model_output = completion.choices[0].message.content
                self.api_key_index += 1
                # 从model_output找到列表
                _ranking_list = re.search(r"\[(.+?)\]", model_output, re.DOTALL)
                _ranking_list = eval(_ranking_list.group(0))
                if len(_ranking_list) > 2:
                    ranking_list = _ranking_list
                    break
            except Exception as e:
                print("-" * 20)
                print(e)
                print("API key error, try again")
                # print(model_output)
                print("-" * 20)
                time.sleep(10)
                continue
        return ranking_list

    def post_process(self, model_output: str):
        """You can post-process the model output here,
        such as extracting the formal proof from the model output."""
        print(model_output)

        solution = re.search(r"# Solution:(.+?)# Problem:", model_output, re.DOTALL)
        problem = re.search(r"# Problem:(.+)", model_output, re.DOTALL)
        if problem:
            problem = problem.group(1)
        else:
            problem = model_output
        if solution:
            solution = solution.group(1)
        else:
            solution = model_output
        # 去除开头末尾的空格和换行符
        problem = problem.strip()
        solution = solution.strip()
        problem = problem.strip("\n")
        solution = solution.strip("\n")
        return problem, solution

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
                + f" All knowledge points in {concept} are {list(conceptDict[qlevel][concept])}. The problem may be based on 1-2 knowledge points, you should according to specific solution to determine them."
            )
        else:
            question_info = "None"
        return question_info

    def get_prompt(self, data):
        try_num = 0
        while try_num < 10:
            try:
                try_num += 1
                para = {
                    "name": data["name"],
                    "formal_proof": data["formal_proof"],
                }
                url = self.base_url + urllib.parse.urlencode(para)
                with urllib.request.urlopen(url) as response:
                    # 读取响应内容
                    para = response.read()
                    # 解析 JSON 响应
                    response = json.loads(para.decode("utf-8"))
                if len(response):
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
                    time.sleep(60)
                    continue
            except:
                time.sleep(60)
                continue

        # ToDo : return a default prompt!
        formal_proof = data["formal_proof"]
        problem_information = self.extact_info(data["name"])
        context = f"""Now, it's your turn! please Try to imitate the linguistic features of solution and problem in the example, such as mathematical terms, problem length, tone, etc., so as to improve the ROUGE-L and BLEU between model predictions and ground truths. You should output the the natural language solution of the problem after the `# Solution:` tag and the problem after the `# Problem:` tag. 
    - Take a deep breath
    - Think step by step
    - I will tip $200
# Problem Information: 
{problem_information}

# Formal solution in Lean 3:
```lean
{formal_proof}
```
"""
        return self.example_prompt_backup, context

    def re_get_prompt(self, data, problem, solution):
        try_num = 0
        while try_num < 10:
            try_num += 1
            para = {
                "name": data["name"],
                "formal_proof": data["formal_proof"],
                "informal_statement": problem,
                "informal_proof": solution,
            }
            url = "http://xxxx/2retrieve_by_problem_solution?"

            url = url + urllib.parse.urlencode(para)
            with urllib.request.urlopen(url) as response:
                # 读取响应内容
                para = response.read()
                # 解析 JSON 响应
                response = json.loads(para.decode("utf-8"))
            if len(response):
                print("成功连接到 API！")
                return response["examples_prompt"], response["context"]
            else:
                print("连接失败，状态码：", response)
                time.sleep(60)
                continue
        return "", ""

    def process_item(self, data):
        examples, context = self.get_prompt(data)

        model_output = self.generate(examples + "\n\n\n" + context)
        problem, solution = self.post_process(model_output)

        return dict(
            name=data["name"],
            formal_proof=data["formal_proof"],
            problem=problem,
            solution=solution,
        )

    def run_backup(self, input_data: str):
        """Run your model on the given input data, and store the
        predictions into the output file."""

        with open(input_data, "r", encoding="utf8") as f:
            datas = json.load(f)
        if not os.path.exists(self.output_file):
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        outputs = []

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_item, item) for item in datas]
            for future in futures:
                outputs.append(future.result())

        with open(self.output_file, "w", encoding="utf8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)

    #! 串行运行代码

    def run(self, input_data: str):
        """Run your model on the given input data, and store the
        predictions into the output file."""

        with open(input_data, "r", encoding="utf8") as f:
            datas = json.load(f)
        if not os.path.exists(self.output_file):
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        outputs = []

        for data in tqdm(datas[:], file=sys.stdout):
            examples, context = self.get_prompt(data)
            model_output = self.generate(examples + "\n\n\n" + context)
            problem, solution = self.post_process(model_output)
            outputs.append(
                dict(
                    name=data["name"],
                    formal_proof=data["formal_proof"],
                    problem=problem,
                    solution=solution,
                )
            )

        with open(self.output_file, "w", encoding="utf8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
