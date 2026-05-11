#!/bin/bash

CHAT_MODEL=Qwen/Qwen3.5-122B-A10B-FP8
CHAT_API_BASE=http://localhost:8000/v1
CHAT_API_KEY=EMPTY

EMBED_MODEL=Qwen/Qwen3-Embedding-8B
EMBED_DIM=1024

TASK_DIRS=(
    ./results/coding
    ./results/writing
)


# # ===============================
# #    Generate descriptions
# # ===============================

# SIMS=(
#     # Format: "simulator_name"  "/path/to/conversations"

#     # Closed-Source Models
#     "gpt-5.4"                 "/path/to/conversations"
#     "gpt-5.4-mini"            "/path/to/conversations"
#     "gpt-5.4-nano"            "/path/to/conversations"
#     "claude_4.5-haiku"        "/path/to/conversations"
#     "gemini-3.1-pro"          "/path/to/conversations"
#     "gemini-3-flash"          "/path/to/conversations"
#     "gemini-3.1-flash-lite"   "/path/to/conversations"

#     # Open-Source Models
#     "qwen3.5-122b-a10b"       "/path/to/conversations"
#     "qwen3.5-35b-a3b"         "/path/to/conversations"
#     "qwen3.5-27b"             "/path/to/conversations"
#     "qwen3.5-9b"              "/path/to/conversations"
#     "qwen3.5-4b"              "/path/to/conversations"
#     "qwen3.5-2b"              "/path/to/conversations"
#     "qwen3.5-0.8b"            "/path/to/conversations"
#     "llama3.3-70b-instruct"   "/path/to/conversations"
#     "llama3.1-8b-instruct"    "/path/to/conversations"
#     "gpt-oss-120b"            "/path/to/conversations"
#     "gpt-oss-20b"             "/path/to/conversations"
#     "gemma-4-31b-it"          "/path/to/conversations"
#     "gemma-4-26b-a4b-it"      "/path/to/conversations"
#     "gemma-4-e2b-it"          "/path/to/conversations"
#     "gemma-4-e4b-it"          "/path/to/conversations"
#     "userlm-8b"               "/path/to/conversations"
#     "humanlm-opinion"         "/path/to/conversations"
# )

# for TASK_DIR in "${TASK_DIRS[@]}"; do
#     echo ""
#     echo "############################################################"
#     echo "# Generating descriptions: ${TASK_DIR}"
#     echo "############################################################"

#     for ((i=0; i<${#SIMS[@]}; i+=2)); do
#         SIMULATOR_NAME="${SIMS[i]}"
#         CONV_FILE="${SIMS[i+1]}"

#         echo ""
#         echo "# Simulator: ${SIMULATOR_NAME}"

#         for FACET in raw_conversation raw_user_utterances raw_intent; do
#             echo "=== Builtin facet: ${FACET} ==="
#             python3 generate_descriptions.py \
#                 --facet ${FACET} \
#                 --input_file "${CONV_FILE}" \
#                 --simulator_name ${SIMULATOR_NAME} \
#                 --output_dir ${TASK_DIR}
#         done

#         for FACET in requests_facet responses_facet context_facet communication_style_facet DAMSL_facet SGD_facet; do
#             echo "=== LLM facet: ${FACET} ==="
#             python3 generate_descriptions.py \
#                 --facet ${FACET} \
#                 --input_file "${CONV_FILE}" \
#                 --simulator_name ${SIMULATOR_NAME} \
#                 --model_name ${CHAT_MODEL} \
#                 --api_base ${CHAT_API_BASE} \
#                 --api_key ${CHAT_API_KEY} \
#                 --max_pending 4096 \
#                 --output_dir ${TASK_DIR}
#         done

#         echo "=== Combined ==="
#         python3 generate_descriptions.py \
#             --facet combined \
#             --input_file "${CONV_FILE}" \
#             --simulator_name ${SIMULATOR_NAME} \
#             --model_name ${CHAT_MODEL} \
#             --api_base ${CHAT_API_BASE} \
#             --output_dir ${TASK_DIR}
#     done
# done


# # ===============================
# #    Embedding representations
# # ===============================

# for TASK_DIR in "${TASK_DIRS[@]}"; do
#     echo ""
#     echo "############################################################"
#     echo "# Embedding: ${TASK_DIR}"
#     echo "############################################################"
#     echo ""
#     python3 embed.py \
#         --model_name ${EMBED_MODEL} \
#         --representations_dir ${TASK_DIR} \
#         --tensor_parallel_size 4 \
#         --chunk_size 1000
# done


# ===============================
#    Evaluate divergence
# ===============================

for TASK_DIR in "${TASK_DIRS[@]}"; do
    if [ -d "${TASK_DIR}" ]; then
        echo ""
        echo "############################################################"
        echo "# Evaluating: ${TASK_DIR}"
        echo "############################################################"
        echo ""
        python3 evaluate.py \
            --representations_dir ${TASK_DIR} \
            --embed_dim ${EMBED_DIM} \
            --k 500
    fi
done