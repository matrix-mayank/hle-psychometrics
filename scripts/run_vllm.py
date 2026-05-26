import modal

# =========================================
# MODAL APP
# =========================================

app = modal.App("hle-vllm")

MAX_MODEL_LEN = 16384
MAX_OUTPUT_TOKENS = 8192
PRINT_RAW_OUTPUTS = False

# =========================================
# REMOTE CONTAINER ENVIRONMENT
# =========================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install(
        "git",
        "build-essential"
    )
    .pip_install(
        "vllm",
        "pandas",
        "huggingface_hub"
    )
    .add_local_file(
        "hle_pilot.csv",
        "/root/project/hle_pilot.csv",
        copy=True
    )
)

# =========================================
# CACHE HUGGINGFACE MODELS
# =========================================

volume = modal.Volume.from_name(
    "hf-cache",
    create_if_missing=True
)

output_volume = modal.Volume.from_name(
    "hle-outputs",
    create_if_missing=True
)

OUTPUT_DIR = "/outputs"

# =========================================
# MAIN EVAL FUNCTION
# =========================================

def run_model_eval(
    model_name,
    model_short,
    input_csv,
    tensor_parallel_size=1,
    dtype=None,
    trust_remote_code=False,
    enable_expert_parallel=False,
    gpu_memory_utilization=0.90,
    max_model_len=None,
    max_output_tokens=None
):

    import os
    import re
    import inspect
    import pandas as pd

    from vllm import LLM, SamplingParams

    # -------------------------------------
    # HF CACHE
    # -------------------------------------

    os.environ["HF_HOME"] = "/cache"
    os.environ["VLLM_USE_FLASHINFER_MOE_FP8"] = "1"

    # -------------------------------------
    # OUTPUT DIRECTORY
    # -------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # -------------------------------------
    # LOAD MODEL
    # -------------------------------------

    print(f"\nLOADING MODEL: {model_name}\n")

    if dtype is None:
        dtype = (
            "bfloat16"
            if "gemma" in model_name.lower()
            else "half"
        )

    effective_max_model_len = max_model_len or MAX_MODEL_LEN
    effective_max_output_tokens = max_output_tokens or MAX_OUTPUT_TOKENS

    llm_kwargs = {
        "model": model_name,
        "tokenizer": model_name,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": effective_max_model_len,
        "tensor_parallel_size": tensor_parallel_size
    }

    llm_signature = inspect.signature(LLM)

    if "trust_remote_code" in llm_signature.parameters:
        llm_kwargs["trust_remote_code"] = trust_remote_code

    if "enable_expert_parallel" in llm_signature.parameters:
        llm_kwargs["enable_expert_parallel"] = enable_expert_parallel

    llm = LLM(**llm_kwargs)

    # -------------------------------------
    # SAMPLING SETTINGS
    # -------------------------------------

    sampling_kwargs = {
        "temperature": 0,
        "max_tokens": effective_max_output_tokens
    }

    if "truncate_prompt_tokens" in inspect.signature(SamplingParams).parameters:
        sampling_kwargs["truncate_prompt_tokens"] = (
            effective_max_model_len - effective_max_output_tokens
        )

    sampling_params = SamplingParams(**sampling_kwargs)

    # -------------------------------------
    # LOAD DATA
    # -------------------------------------

    csv_path = f"/root/project/{input_csv}"

    df = pd.read_csv(csv_path)

    item_id_column = (
        "id"
        if "id" in df.columns
        else "item_id"
        if "item_id" in df.columns
        else None
    )

    # -------------------------------------
    # BUILD PROMPTS
    # -------------------------------------

    prompt_prefix = """
Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}

"""

    tokenizer = llm.get_tokenizer()
    prompt_prefix_ids = tokenizer.encode(prompt_prefix)
    max_prompt_tokens = effective_max_model_len - effective_max_output_tokens
    available_question_tokens = max_prompt_tokens - len(prompt_prefix_ids)

    model_inputs = []
    truncated_count = 0

    def encode_question(question):
        try:
            return tokenizer.encode(
                question,
                add_special_tokens=False
            )
        except TypeError:
            return tokenizer.encode(question)

    for _, row in df.iterrows():

        question = str(row["question"])
        question_ids = encode_question(question)

        if len(question_ids) > available_question_tokens:
            question_ids = question_ids[-available_question_tokens:]
            truncated_count += 1

        model_inputs.append({
            "prompt_token_ids": prompt_prefix_ids + question_ids
        })

    if truncated_count:
        print(
            f"TRUNCATED {truncated_count} PROMPTS TO FIT "
            f"{max_prompt_tokens} INPUT TOKENS"
        )

    # -------------------------------------
    # GENERATE
    # -------------------------------------

    print("STARTING GENERATION\n")

    outputs = llm.generate(
        model_inputs,
        sampling_params
    )

    # -------------------------------------
    # PROCESS OUTPUTS
    # -------------------------------------

    results = []

    for idx, output in enumerate(outputs):

        response_text = (
            output.outputs[0]
            .text
            .strip()
        )

        # ---------------------------------
        # PARSE ANSWER
        # ---------------------------------

        answer_patterns = [
            r"(?:\*\*\s*)?Answer\s*(?:\*\*)?\s*:\s*"
            r"(?:\*\*)?\s*(?:option\s+|choice\s+|letter\s+)?"
            r"(?:is\s+)?(?:\*\*)?\s*\(?([A-Z])\)?"
            r"(?:\*\*)?\s*(?=$|[\s\.,;:\)\]])",
            r"(?:\*\*\s*)?(?:Final\s+)?Answer\s*(?:\*\*)?\s+"
            r"(?:is\s+)?(?:\*\*)?\s*\(?([A-Z])\)?"
            r"(?:\*\*)?\s*(?=$|[\s\.,;:\)\]])",
            r"(?:answer|choice|option|letter|final\s+answer)"
            r"[^.\n]{0,80}?\bis\s+(?:\*\*)?\s*\(?([A-Z])\)?"
            r"(?:\*\*)?\s*(?=$|[\s\.,;:\)\]])"
        ]

        answer_match = None

        for answer_pattern in answer_patterns:
            answer_match = re.search(
                answer_pattern,
                response_text,
                re.IGNORECASE
            )

            if answer_match:
                break

        answer = (
            answer_match.group(1)
            if answer_match else ""
        )

        # ---------------------------------
        # PARSE CONFIDENCE
        # ---------------------------------

        confidence_match = re.search(
            r"(?:\*\*\s*)?Confidence\s*(?:\*\*)?\s*:\s*([0-9]+)",
            response_text,
            re.IGNORECASE
        )

        confidence = (
            int(confidence_match.group(1))
            if confidence_match else None
        )

        print("\n=================================")
        print(f"ITEM {idx}")
        print("=================================")
        print(f"Answer: {answer}")
        print(f"Confidence: {confidence}")

        if PRINT_RAW_OUTPUTS:
            print(response_text)

        print("=================================\n")

        item_id = (
            df.iloc[idx][item_id_column]
            if item_id_column else idx
        )

        ground_truth = (
            df.iloc[idx]["answer"]
            if "answer" in df.columns else ""
        )

        results.append({
            "model": model_short,
            "item_id": item_id,
            "response": answer,
            "confidence": confidence,
            "raw_output": response_text,
            "ground_truth": ground_truth,
            "finish_reason": getattr(output.outputs[0], "finish_reason", None),
            "stop_reason": getattr(output.outputs[0], "stop_reason", None),
            "generated_token_count": len(
                getattr(output.outputs[0], "token_ids", []) or []
            )
        })

    # -------------------------------------
    # SAVE LONG FORMAT
    # -------------------------------------

    results_df = pd.DataFrame(results)

    long_path = (
        f"{OUTPUT_DIR}/"
        f"{model_short}_responses_long.csv"
    )

    results_df.to_csv(
        long_path,
        index=False
    )

    # -------------------------------------
    # SAVE RESPONSE MATRIX
    # -------------------------------------

    matrix_df = results_df.pivot(
        index="model",
        columns="item_id",
        values="response"
    )

    matrix_path = (
        f"{OUTPUT_DIR}/"
        f"{model_short}_responses_matrix.csv"
    )

    matrix_df.to_csv(matrix_path)
    output_volume.commit()

    # -------------------------------------
    # PRINT SUMMARY
    # -------------------------------------

    print("\nLONG FORMAT:")
    print(results_df)

    print("\nRESPONSE MATRIX:")
    print(matrix_df)

    return {
        "long_format": long_path,
        "matrix_format": matrix_path
    }

# =========================================
# SMALL MODEL GPU FUNCTION
# =========================================

@app.function(
    gpu="A100",
    secrets=[
        modal.Secret.from_name("huggingface")
    ],
    image=image,
    volumes={
        "/cache": volume,
        OUTPUT_DIR: output_volume
    },
    timeout=60 * 60 * 12
)
def run_small_model(
    model_name,
    model_short,
    input_csv="hle_pilot.csv",
    tensor_parallel_size=1,
    dtype=None,
    trust_remote_code=False,
    enable_expert_parallel=False,
    gpu_memory_utilization=0.90,
    max_model_len=None,
    max_output_tokens=None
):

    return run_model_eval(
        model_name,
        model_short,
        input_csv,
        tensor_parallel_size,
        dtype,
        trust_remote_code,
        enable_expert_parallel,
        gpu_memory_utilization,
        max_model_len,
        max_output_tokens
    )

# =========================================
# LARGE MODEL GPU FUNCTION
# =========================================

@app.function(
    gpu="A100-80GB",
    secrets=[
    modal.Secret.from_name("huggingface")
    ],
    image=image,
    volumes={
        "/cache": volume,
        OUTPUT_DIR: output_volume
    },
    timeout=60 * 60 * 12
)
def run_large_model(
    model_name,
    model_short,
    input_csv="hle_pilot.csv",
    tensor_parallel_size=1,
    dtype=None,
    trust_remote_code=False,
    enable_expert_parallel=False,
    gpu_memory_utilization=0.90,
    max_model_len=None,
    max_output_tokens=None
):

    return run_model_eval(
        model_name,
        model_short,
        input_csv,
        tensor_parallel_size,
        dtype,
        trust_remote_code,
        enable_expert_parallel,
        gpu_memory_utilization,
        max_model_len,
        max_output_tokens
    )

# =========================================
# XL MODEL GPU FUNCTION
# =========================================

@app.function(
    gpu="H100:2",
    secrets=[
        modal.Secret.from_name("huggingface")
    ],
    image=image,
    volumes={
        "/cache": volume,
        OUTPUT_DIR: output_volume
    },
    timeout=60 * 60 * 12
)
def run_xl_model(
    model_name,
    model_short,
    input_csv="hle_pilot.csv",
    tensor_parallel_size=2,
    dtype=None,
    trust_remote_code=False,
    enable_expert_parallel=False,
    gpu_memory_utilization=0.90,
    max_model_len=None,
    max_output_tokens=None
):

    return run_model_eval(
        model_name,
        model_short,
        input_csv,
        tensor_parallel_size,
        dtype,
        trust_remote_code,
        enable_expert_parallel,
        gpu_memory_utilization,
        max_model_len,
        max_output_tokens
    )

# =========================================
# B200 4-GPU MODEL FUNCTION
# =========================================

@app.function(
    gpu="B200:4",
    secrets=[
        modal.Secret.from_name("huggingface")
    ],
    image=image,
    volumes={
        "/cache": volume,
        OUTPUT_DIR: output_volume
    },
    timeout=60 * 60 * 12
)
def run_b200_4_model(
    model_name,
    model_short,
    input_csv="hle_pilot.csv",
    tensor_parallel_size=4,
    dtype=None,
    trust_remote_code=False,
    enable_expert_parallel=False,
    gpu_memory_utilization=0.90,
    max_model_len=None,
    max_output_tokens=None
):

    return run_model_eval(
        model_name,
        model_short,
        input_csv,
        tensor_parallel_size,
        dtype,
        trust_remote_code,
        enable_expert_parallel,
        gpu_memory_utilization,
        max_model_len,
        max_output_tokens
    )

# =========================================
# H200 8-GPU MODEL FUNCTION
# =========================================

@app.function(
    gpu="H200:8",
    secrets=[
        modal.Secret.from_name("huggingface")
    ],
    image=image,
    volumes={
        "/cache": volume,
        OUTPUT_DIR: output_volume
    },
    timeout=60 * 60 * 12
)
def run_h200_8_model(
    model_name,
    model_short,
    input_csv="hle_pilot.csv",
    tensor_parallel_size=8,
    dtype=None,
    trust_remote_code=False,
    enable_expert_parallel=False,
    gpu_memory_utilization=0.90,
    max_model_len=None,
    max_output_tokens=None
):

    return run_model_eval(
        model_name,
        model_short,
        input_csv,
        tensor_parallel_size,
        dtype,
        trust_remote_code,
        enable_expert_parallel,
        gpu_memory_utilization,
        max_model_len,
        max_output_tokens
    )

# =========================================
# LOCAL ENTRYPOINT
# =========================================

@app.local_entrypoint()
def main():

    small_models = [

        # {
        #     "model_name": "allenai/OLMo-2-1124-7B-Instruct",
        #     "model_short": "olmo2_7b_4k2k",
        #     "dtype": "bfloat16",
        #     "max_model_len": 4096,
        #     "max_output_tokens": 2048
        # },
        # {
        #     "model_name": "tiiuae/Falcon3-10B-Instruct",
        #     "model_short": "falcon3_10b_16k8k",
        #     "dtype": "half"
        # },
        {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "model_short": "llama31_8b_16k8k",
            "dtype": "bfloat16"
        },
        # {
        #     "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        #     "model_short": "mistral7b_16k8k",
        #     "dtype": "half"
        # }
        # {
        #     "model_name": "google/gemma-2-9b-it",
        #     "model_short": "gemma9b_8k4k",
        #     "dtype": "bfloat16",
        #     "max_model_len": 8192,
        #     "max_output_tokens": 4096
        # },
        # {
        #     "model_name": "Qwen/Qwen2.5-7B-Instruct",
        #     "model_short": "qwen7b_16k8k",
        #     "dtype": "half"
        # }
    ]

    large_models = [

        # {
        #     "model_name": "microsoft/phi-4",
        #     "model_short": "phi4_16k8k",
        #     "dtype": "half"
        # },
        # {
        #     "model_name": "google/gemma-3-27b-it",
        #     "model_short": "gemma27b_16k8k",
        #     "dtype": "bfloat16"
        # }
    ]

    xl_models = [

        # {
        #     "model_name": "NovaSky-AI/Sky-T1-32B-Preview",
        #     "model_short": "sky_t1_32b_16k8k",
        #     "tensor_parallel_size": 2,
        #     "dtype": "bfloat16"
        # },
        # {
        #     "model_name": "Qwen/QwQ-32B",
        #     "model_short": "qwq_32b_16k8k",
        #     "tensor_parallel_size": 2,
        #     "dtype": "bfloat16"
        # }
    ]

    b200_4_models = [
        # {
        #     "model_name": "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        #     "model_short": "qwen3_235b_thinking",
        #     "tensor_parallel_size": 4,
        #     "dtype": "auto",
        #     "trust_remote_code": True,
        #     "enable_expert_parallel": False
        # }
    ]

    h200_8_models = [

        # {
        #     "model_name": "deepseek-ai/DeepSeek-R1",
        #     "model_short": "deepseek_r1",
        #     "tensor_parallel_size": 8,
        #     "dtype": "auto",
        #     "trust_remote_code": True,
        #     "enable_expert_parallel": True
        # },
        # {
        #     "model_name": "deepseek-ai/DeepSeek-V3",
        #     "model_short": "deepseek_v3",
        #     "tensor_parallel_size": 8,
        #     "dtype": "auto",
        #     "trust_remote_code": True,
        #     "enable_expert_parallel": True
        # },
        # {
        #     "model_name": "deepseek-ai/DeepSeek-R1-Zero",
        #     "model_short": "deepseek_r1_zero",
        #     "tensor_parallel_size": 8,
        #     "dtype": "auto",
        #     "trust_remote_code": True,
        #     "enable_expert_parallel": True
        # }
    ]

    # -------------------------------------
    # RUN SMALL MODELS
    # -------------------------------------

    for model in small_models:

        print("\n=================================")
        print(f"RUNNING SMALL MODEL:")
        print(model["model_short"])
        print("=================================\n")

        run_small_model.remote(

            model_name=model["model_name"],

            model_short=model["model_short"],

            input_csv="hle_pilot.csv",

            tensor_parallel_size=model.get("tensor_parallel_size", 1),

            dtype=model.get("dtype"),

            trust_remote_code=model.get("trust_remote_code", False),

            enable_expert_parallel=model.get("enable_expert_parallel", False),

            gpu_memory_utilization=model.get("gpu_memory_utilization", 0.90),

            max_model_len=model.get("max_model_len"),

            max_output_tokens=model.get("max_output_tokens")
        )

    # -------------------------------------
    # RUN LARGE MODELS
    # -------------------------------------

    for model in large_models:

        print("\n=================================")
        print(f"RUNNING LARGE MODEL:")
        print(model["model_short"])
        print("=================================\n")

        run_large_model.remote(

            model_name=model["model_name"],

            model_short=model["model_short"],

            input_csv="hle_pilot.csv",

            tensor_parallel_size=model.get("tensor_parallel_size", 1),

            dtype=model.get("dtype"),

            trust_remote_code=model.get("trust_remote_code", False),

            enable_expert_parallel=model.get("enable_expert_parallel", False),

            gpu_memory_utilization=model.get("gpu_memory_utilization", 0.90),

            max_model_len=model.get("max_model_len"),

            max_output_tokens=model.get("max_output_tokens")
        )

    # -------------------------------------
    # RUN XL MODELS
    # -------------------------------------

    for model in xl_models:

        print("\n=================================")
        print(f"RUNNING XL MODEL:")
        print(model["model_short"])
        print("=================================\n")

        run_xl_model.remote(

            model_name=model["model_name"],

            model_short=model["model_short"],

            input_csv="hle_pilot.csv",

            tensor_parallel_size=model.get("tensor_parallel_size", 2),

            dtype=model.get("dtype"),

            trust_remote_code=model.get("trust_remote_code", False),

            enable_expert_parallel=model.get("enable_expert_parallel", False),

            gpu_memory_utilization=model.get("gpu_memory_utilization", 0.90),

            max_model_len=model.get("max_model_len"),

            max_output_tokens=model.get("max_output_tokens")
        )

    # -------------------------------------
    # RUN B200 4-GPU MODELS
    # -------------------------------------

    for model in b200_4_models:

        print("\n=================================")
        print(f"RUNNING B200 4-GPU MODEL:")
        print(model["model_short"])
        print("=================================\n")

        run_b200_4_model.remote(

            model_name=model["model_name"],

            model_short=model["model_short"],

            input_csv="hle_pilot.csv",

            tensor_parallel_size=model.get("tensor_parallel_size", 4),

            dtype=model.get("dtype"),

            trust_remote_code=model.get("trust_remote_code", False),

            enable_expert_parallel=model.get("enable_expert_parallel", False),

            gpu_memory_utilization=model.get("gpu_memory_utilization", 0.90),

            max_model_len=model.get("max_model_len"),

            max_output_tokens=model.get("max_output_tokens")
        )

    # -------------------------------------
    # RUN H200 8-GPU MODELS
    # -------------------------------------

    for model in h200_8_models:

        print("\n=================================")
        print(f"RUNNING H200 8-GPU MODEL:")
        print(model["model_short"])
        print("=================================\n")

        run_h200_8_model.remote(

            model_name=model["model_name"],

            model_short=model["model_short"],

            input_csv="hle_pilot.csv",

            tensor_parallel_size=model.get("tensor_parallel_size", 8),

            dtype=model.get("dtype"),

            trust_remote_code=model.get("trust_remote_code", False),

            enable_expert_parallel=model.get("enable_expert_parallel", False),

            gpu_memory_utilization=model.get("gpu_memory_utilization", 0.90),

            max_model_len=model.get("max_model_len"),

            max_output_tokens=model.get("max_output_tokens")
        )
