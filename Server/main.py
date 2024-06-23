from fastapi import FastAPI
import json
import openai
import re
import logging
import torch
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import LLMEmbedder
import numpy as np
from func_timeout import func_set_timeout, FunctionTimedOut
from MUSTARD.leaven.src.lean_server import LeanEnv
import time
import random
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from sentence_transformers.quantization import quantize_embeddings

openai.api_key = "xxxx"
openai.base_url = "https://api.chatanywhere.com.cn/"


ModelDir = r"1-1-Formalization/new_training_data/"

model = "gpt-3.5-turbo-0125"
temperature = 0.0

app = FastAPI()

bge = BGEM3FlagModel(
    "./bge-m3",
    use_fp16=True,
    device="cuda",
    pooling_method="mean",
)
# mxbai = SentenceTransformer("mxbai-large-v1", device='cuda')
vector_library_1 = torch.load(ModelDir + r"vector_library.pt")
vector_library_1_code = torch.load(ModelDir + r"vector_library_code.pt")
vector_library_1_llmemb = torch.load(
    "1-1-Formalization/new_training_data/vector_library_code_llmemb.pt"
)
vector_library_2 = torch.load(
    r"1-2-Informalization/task1-2_public_data/vector_library.pt"
)
# vector_library_2_mxbai = torch.load(
#         r"1-2-Informalization/task1-2_public_data/vector_library_mxbai.pt"
#     )


@app.get("/")
def read_root():
    return {"Hello": "World"}


def get_embedding(text, model="text-embedding-ada-002"):
    try_num = 0
    while try_num < 10:
        try_num += 1
        text = text.replace("\n", " ")
        client = OpenAI(
            api_key="sk-miuLp6NefxjhNhiTeitQWua4gBkbsCDHg9HiP34Rp1JH45XK",
            base_url="https://api.chatanywhere.tech/v1",
        )
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    return None


def retrieval_questions(text1, text2):
    with torch.no_grad():
        embeddings1 = bge.encode(
            text1,
            batch_size=12,
            max_length=8192,
        )["dense_vecs"]
        embeddings2 = bge.encode(
            text2,
            batch_size=12,
            max_length=8192,
        )["dense_vecs"]
        embeddings1 = torch.from_numpy(np.array(embeddings1)).to("cuda")
        embeddings2 = torch.from_numpy(np.array(embeddings2)).to("cuda")

        informal_statements_embedding = vector_library_1[
            "informal_statements_embedding"
        ].to("cuda")
        informal_proofs_embedding = vector_library_1["informal_proofs_embedding"].to(
            "cuda"
        )
        # 计算相似度
        sim1 = embeddings1 @ informal_statements_embedding.T
        sim2 = embeddings2 @ informal_proofs_embedding.T

        # 选择top-k索引及对应的相似度值
        k = 4
        topk_values1, selected_idxs1 = torch.topk(sim1, k=k)
        topk_values2, selected_idxs2 = torch.topk(sim2, k=k)

        # 将索引转换为列表
        selected_idxs1 = selected_idxs1.squeeze(0).tolist()
        selected_idxs2 = selected_idxs2.squeeze(0).tolist()

        # 将相似度值转换为列表
        topk_values1 = topk_values1.squeeze(0).tolist()
        topk_values2 = topk_values2.squeeze(0).tolist()

        # 合并索引和相似度值
        combined_idxs = selected_idxs1 + selected_idxs2
        combined_values = topk_values1 + topk_values2

        # 使用字典来保留索引和对应的相似度值
        idx_value_pairs = list(zip(combined_idxs, combined_values))

        # 去重并按照相似度值降序排序
        idx_value_pairs = list(set(idx_value_pairs))
        idx_value_pairs.sort(key=lambda x: x[1], reverse=True)

        # 提取排序后的索引
        sorted_idxs = [idx for idx, _ in idx_value_pairs]
        sorted_idxs = list(set(sorted_idxs))
        # 输出排序后的索引
        # print(sorted_idxs)

    return sorted_idxs


def retrieval_questions_by_code(text1, text2):

    with torch.no_grad():
        embeddings1 = bge.encode(
            text1,
            batch_size=12,
            max_length=8192,
        )["dense_vecs"]
        embeddings2 = bge.encode(
            text2,
            batch_size=12,
            max_length=8192,
        )["dense_vecs"]
        embeddings1 = torch.from_numpy(np.array(embeddings1)).to("cuda")
        embeddings2 = torch.from_numpy(np.array(embeddings2)).to("cuda")

        informal_proofs_embedding = vector_library_1_code[
            "informal_proofs_embedding"
        ].to("cuda")
        formal_proofs_embedding = vector_library_1_code["formal_proofs_embedding"].to(
            "cuda"
        )

        # 计算相似度
        sim1 = embeddings1 @ informal_proofs_embedding.T
        sim2 = embeddings2 @ formal_proofs_embedding.T

        # 选择top-k索引及对应的相似度值
        k = 4
        topk_values1, selected_idxs1 = torch.topk(sim1, k=k)
        topk_values2, selected_idxs2 = torch.topk(sim2, k=k)

        # 将索引转换为列表
        selected_idxs1 = selected_idxs1.squeeze(0).tolist()
        selected_idxs2 = selected_idxs2.squeeze(0).tolist()

        # 将相似度值转换为列表
        topk_values1 = topk_values1.squeeze(0).tolist()
        topk_values2 = topk_values2.squeeze(0).tolist()

        # 合并索引和相似度值
        combined_idxs = selected_idxs1 + selected_idxs2
        combined_values = topk_values1 + topk_values2

        # 使用字典来保留索引和对应的相似度值
        idx_value_pairs = list(zip(combined_idxs, combined_values))

        # 去重并按照相似度值降序排序
        idx_value_pairs = list(set(idx_value_pairs))
        idx_value_pairs.sort(key=lambda x: x[1], reverse=True)

        # 提取排序后的索引
        sorted_idxs = [idx for idx, _ in idx_value_pairs]
        sorted_idxs = list(set(sorted_idxs))
        # 输出排序后的索引
        # print(sorted_idxs)

    return sorted_idxs


def retrieval_questions_use_LLMEmbedder(text1, text2):
    llm_embedder = LLMEmbedder("llm-embedder", use_fp16=False, pooling_method="mean")
    with torch.no_grad():
        task = "icl"
        embeddings1 = llm_embedder.encode_queries(text1, task=task)
        embeddings2 = llm_embedder.encode_queries(text2, task=task)
        embeddings1 = torch.from_numpy(np.array(embeddings1)).to("cuda")
        embeddings2 = torch.from_numpy(np.array(embeddings2)).to("cuda")

        informal_statements_embedding = vector_library_1_llmemb[
            "informal_statements_embedding"
        ].to("cuda")
        informal_proofs_embedding = vector_library_1_llmemb[
            "informal_proofs_embedding"
        ].to("cuda")
        # 计算相似度
        sim1 = embeddings1 @ informal_statements_embedding.T
        sim2 = embeddings2 @ informal_proofs_embedding.T

        # 选择top-k索引及对应的相似度值
        k = 4
        topk_values1, selected_idxs1 = torch.topk(sim1, k=k)
        topk_values2, selected_idxs2 = torch.topk(sim2, k=k)

        # 将索引转换为列表
        selected_idxs1 = selected_idxs1.squeeze(0).tolist()
        selected_idxs2 = selected_idxs2.squeeze(0).tolist()

        # 将相似度值转换为列表
        topk_values1 = topk_values1.squeeze(0).tolist()
        topk_values2 = topk_values2.squeeze(0).tolist()

        # 合并索引和相似度值
        combined_idxs = selected_idxs1 + selected_idxs2
        combined_values = topk_values1 + topk_values2

        # 使用字典来保留索引和对应的相似度值
        idx_value_pairs = list(zip(combined_idxs, combined_values))

        # 去重并按照相似度值降序排序
        idx_value_pairs = list(set(idx_value_pairs))
        idx_value_pairs.sort(key=lambda x: x[1], reverse=True)

        # 提取排序后的索引
        sorted_idxs = [idx for idx, _ in idx_value_pairs]
        sorted_idxs = list(set(sorted_idxs))
        # 输出排序后的索引
        # print(sorted_idxs)

    return sorted_idxs


def retrieval_questions2(text1, text2):
    with torch.no_grad():
        text1_embeddings = bge.encode(
            [text1],
            batch_size=12,
            max_length=8192,
        )["dense_vecs"]
        text1_embeddings = torch.from_numpy(np.array(text1_embeddings)).to("cuda")
        formal_proofs_embedding = vector_library_2["formal_proofs_embedding"].to("cuda")
        # 计算相似度
        sim1 = text1_embeddings @ formal_proofs_embedding.T

        # 选择top-k索引及对应的相似度值
        topk_values1, selected_idxs1 = torch.topk(sim1, k=8)
        selected_idxs1 = selected_idxs1.squeeze(0).tolist()
        return selected_idxs1
        # 取出对应的code
        # training_data = r"1-2-Informalization/task1-2_public_data/training_data.json"
        # with open(training_data, "r", encoding="utf8") as f:
        #     training_data = json.load(f)
        # text_pairs = []
        # for idx in selected_idxs1:
        #     text_pairs.append([text1, training_data[idx]["formal_proof"]])
        # sim2 = bge.compute_score(text_pairs,
        #                   weights_for_different_modes=[0.4, 0.2, 0.4])['colbert+sparse+dense']
        # sim2 = torch.from_numpy(np.array(sim2))
        # topk_values2, selected_idxs2 = torch.topk(sim2, k=8)

        # return selected_idxs2.squeeze(0).tolist()


def retrieval_questions2_by_mxbai(text1, text2):
    with torch.no_grad():
        text1_embeddings = mxbai.encode(
            ["Represent this sentence for searching relevant passages: " + text1],
            device="cuda",
        )
        text1_embeddings = torch.from_numpy(np.array(text1_embeddings)).to("cuda")
        formal_proofs_embedding = vector_library_2_mxbai["formal_proofs_embedding"].to(
            "cuda"
        )
        # 计算相似度
        # sim1 = text1_embeddings @ formal_proofs_embedding.T
        sim1 = cos_sim(text1_embeddings, formal_proofs_embedding)
        # 选择top-k索引及对应的相似度值
        topk_values1, selected_idxs1 = torch.topk(sim1, k=12)

        return selected_idxs1.squeeze(0).tolist()


def retrieval_questions2_by_openaiembedding(text1, text2):
    with torch.no_grad():
        text1_embeddings = bge.encode(
            [text1],
            batch_size=12,
            max_length=8192,
        )["dense_vecs"]
        text1_embeddings = torch.from_numpy(np.array(text1_embeddings)).to("cuda")
        formal_proofs_embedding = vector_library_2["formal_proofs_embedding"].to("cuda")
        # 计算相似度
        sim1 = text1_embeddings @ formal_proofs_embedding.T

        # 选择top-k索引及对应的相似度值
        topk_values1, selected_idxs1 = torch.topk(sim1, k=8)

        return selected_idxs1.squeeze(0).tolist()


def extact_info(name):
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


def get_prompt(examples_idx, query, informal_statement, informal_proof):
    training_data = ModelDir + r"training_data.json"
    training_label = ModelDir + r"training_label.json"

    with open(training_data, "r", encoding="utf8") as f:
        training_data = json.load(f)
    with open(training_label, "r", encoding="utf8") as f:
        training_label = json.load(f)

    examples = """Here are some examples you can refer to:\n"""
    for idx in examples_idx:
        # name = training_data[idx]["name"]
        # problem_information = extact_info(name)
        problem = training_data[idx]["informal_statement"]
        solution = training_data[idx]["informal_proof"]
        lean_code = training_label[idx]["formal_proof"]
        # examples += f"# Problem Information:\n{problem_information}\n"
        examples += f"# Problem:\n{problem}\n# Solution:\n{solution}\n"
        examples += f"# Formal solution in Lean 3:\n```lean\n{lean_code}\n```\n---\n"

    # problem_information = extact_info(query)
    context = f"""Now, It's your turn! Please Try to imitate the linguistic features of Lean 3 in the example, such as mathematical terms, variable naming, comments, code style etc., so as to improve the ROUGE-L and BLEU between model prediction and ground truth.
    - Take a deep breath
    - Think step by step 
    - I will tip $200

# Problem:
{informal_statement}
# Solution:
{informal_proof}
# Formal solution in Lean 3:

"""
    logging.info(f"get_prompt: {examples + context}")
    return examples, context


def get_prompt2(examples_idx, query, formal_proof):
    training_data = r"1-2-Informalization/task1-2_public_data/training_data.json"
    training_label = r"1-2-Informalization/task1-2_public_data/training_label.json"

    with open(training_data, "r", encoding="utf8") as f:
        training_data = json.load(f)
    with open(training_label, "r", encoding="utf8") as f:
        training_label = json.load(f)

    examples = """Here are some examples you can refer to：\n"""
    for idx in examples_idx:
        problem = training_label[idx]["informal_statement"]
        solution = training_label[idx]["informal_proof"]
        lean_code = training_data[idx]["formal_proof"]
        name = training_data[idx]["name"]
        problem_information = extact_info(name)
        examples += f"# Problem Information:\n{problem_information}\n"
        examples += f"# Formal solution in Lean 3:\n```lean\n{lean_code}\n```\n"
        examples += f"# Solution:\n{solution}\n# Problem:\n{problem}\n---\n"

    problem_information = extact_info(query)
    context = f"""Now, it's your turn! Please Try to imitate the linguistic features of solution and problem in the example, such as mathematical terms, problem length, tone, etc., so as to improve the ROUGE-L and BLEU between model predictions and ground truths. You should output the the natural language solution of the problem after the `# Solution:` tag and the problem after the `# Problem:` tag. 
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
    return examples, context


def ChatGPT(prompt):

    # print("#############################################################################################################")
    # print("ChatGPT")
    # print("#############################################################################################################")
    messages = [
        {"role": "user", "content": prompt},
    ]
    try_num = 0
    while True:
        try_num += 1
        if try_num > 10:
            return ""
        try:
            completion = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=0.7,
                frequency_penalty=0.0,
            )
            model_output = completion.choices[0].message.content
            code = re.search(r"```lean(.+?)```", model_output, re.DOTALL)
            if code and len(code.group(1).strip()):
                break
        except Exception as e:
            # print(e)
            logging.error(e)
            continue
    return completion.choices[0].message.content


def post_process(model_output: str):
    try:
        formal_proof = re.search(r"```lean(.+?)```", model_output, re.DOTALL)

        formal_proof = formal_proof.group(1).strip()
        lean_code = "\n".join(formal_proof.strip().split("\n"))
        lean_code = re.sub(
            pattern=r"line [0-9]* ", repl="", string=lean_code
        )  # remove line *
    except Exception as e:
        # print(e)
        logging.error(e)
        lean_code = "none"
    return lean_code


@app.get("/retrieve")
async def retrieve(name: str, informal_statement: str, informal_proof: str):
    examples_idx = retrieval_questions([informal_statement], [informal_proof])
    examples_prompt, context = get_prompt(
        examples_idx, name, informal_statement, informal_proof
    )
    api_key_pool = []
    return {
        "examples_idx": examples_idx,
        "examples_prompt": examples_prompt,
        "context": context,
        "api_key_pool": api_key_pool,
    }


@app.get("/retrievecode")
async def retrievecode(
    name: str, informal_statement: str, informal_proof: str, formal_proof: str
):
    examples_idx = retrieval_questions_by_code([informal_proof], [formal_proof])
    examples_prompt, context = get_prompt(
        examples_idx, name, informal_statement, informal_proof
    )
    return {"examples_idx": examples_idx, "examples_prompt": examples_prompt}


@app.get("/2retrieve")
async def retrieve2(name: str, formal_proof: str):
    examples_idx = retrieval_questions2(formal_proof, extact_info(name))
    examples, context = get_prompt2(examples_idx, name, formal_proof)
    api_key_pool = []

    return {
        "examples_idx": examples_idx,
        "examples_prompt": examples,
        "context": context,
        "api_key_pool": api_key_pool,
    }


@app.get("/2retrieve_by_problem_solution")
async def retrieve2_by_problem_solution(
    name: str, formal_proof: str, informal_statement: str, informal_proof: str
):
    examples_idx = retrieval_questions([informal_statement], [informal_proof])
    examples, context = get_prompt2(examples_idx, name, formal_proof)
    return {
        "examples_idx": examples_idx,
        "examples_prompt": examples,
        "context": context,
    }


async def verify_lean_file(lean_code):
    server = LeanEnv()
    infos = server.verify_lean_file(lean_code)
    server.close()
    return infos


@app.get("/verify_lean")
async def verify_lean(leancode: str):
    try:
        start_time = time.time()
        infos = await verify_lean_file(leancode)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
        print(infos)
    # except FunctionTimedOut:
    #     print("Function execution timed out.")
    #     return {}
    except Exception as e:
        print("An error occurred:", e)
        return {}
    return infos
