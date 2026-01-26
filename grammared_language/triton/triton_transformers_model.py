# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from ast import Load
import logging
import os
import json
from xml.parsers.expat import model

import numpy as np
import torch
import transformers
import triton_python_backend_utils as pb_utils
from grammared_language.utils.config_parser import get_model_config, BaseModelConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG) 


DEFAULT_MODEL_BACKEND = "transformers"
def load_pipeline_from_config(model_config:BaseModelConfig, task="text2text-generation", backend:str="transformers", fallback:bool=True, **kwargs):
    """Load a HuggingFace model based on the provided configuration.

    Args:
        model_config (BaseModelConfig): Model configuration

    Returns:
        model: Loaded HuggingFace model.
    """
    # Extract model parameters from configuration
    default_hf_model = "HuggingFaceTB/SmolLM2-135M"
    if model_config.serving_config.pretrained_model_name_or_path:
        hf_model = model_config.serving_config.pretrained_model_name_or_path
    else:
        hf_model = default_hf_model
    logger.warning(f"Loading model: {hf_model} with backend: {backend}")

    try:
        if backend == "ort":
            from optimum import onnxruntime
            pipeline = onnxruntime.pipeline(task=task, model=hf_model, **kwargs)
        else:
            pipeline = transformers.pipeline(task=task, model=hf_model, **kwargs)
    except Exception as e:
        if fallback and backend != DEFAULT_MODEL_BACKEND:
            logger.warning(f"Failed to load model {hf_model}: {str(e)} using backend: {backend}. Falling back to default model backend {DEFAULT_MODEL_BACKEND}.")
            pipeline = transformers.pipeline(task=task, model=hf_model, **kwargs)
        else:
            raise RuntimeError(f"Failed to load model {hf_model}: {str(e)}")
    return pipeline


class TritonTransformersPythonModel:
    PIPELINE_TASK = "text-generation"
    DEFAULT_HF_MODEL = "HuggingFaceTB/SmolLM2-135M"
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        self.model_params = self.model_config.get("parameters", {})
        self.grammared_language_model_config = json.loads(
            self.model_config.get('parameters', {}).get('grammared_language_model_config', {}).get('string_value', "{}")
        )
        self.grammared_language_model_config = get_model_config(self.model_config.get("name", "transfomers_model"), self.grammared_language_model_config)

        default_max_gen_length = "30"
        self.max_output_length = self.grammared_language_model_config.model_inference_config.max_length or int(default_max_gen_length)
        # Check for user-specified model name in model config parameters
        hf_model = self.model_params.get("pretrained_model_name_or_path", {}).get(
            "string_value", self.DEFAULT_HF_MODEL
        )

        logger.warning(f"Loading HuggingFace model: {hf_model}...")
        logger.warning(f"Loading grammared_language_model_config: {self.grammared_language_model_config}...")

        # Get model instance device configuration
        model_device_type = args['model_instance_kind']
        model_instance_device_id = args['model_instance_device_id']
        if self.grammared_language_model_config.serving_config.device == 'cpu':
            self.device = 'cpu'
        elif self.grammared_language_model_config.serving_config.device == 'cuda':
            self.device = f'cuda:{model_instance_device_id}'
        else:
            self.device = 'auto'

        # # Assume tokenizer available for same model
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
        # self.pipeline = transformers.pipeline(
        #     self.PIPELINE_TASK,
        #     model=hf_model,
        #     # torch_dtype=torch.float16,
        #     tokenizer=self.tokenizer,
        #     device_map="auto",
        # )

        self.pipeline = load_pipeline_from_config(
            self.grammared_language_model_config,
            task=self.PIPELINE_TASK,
            backend=self.grammared_language_model_config.serving_config.backend or DEFAULT_MODEL_BACKEND,
            fallback=True,
            device=self.device,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get input tensor with shape [batch_size, 1]
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            input_data = input_tensor.as_numpy()
            
            # Process all items in the batch
            prompts = []
            for i in range(len(input_data)):
                prompt = input_data[i][0].decode("utf-8")
                prompts.append(prompt)
                logger.warning(f"Batch item {i}: {prompt}")
            
            response = self.generate_batch(prompts)
            responses.append(response)

        return responses

    def generate_batch(self, prompts):
        """Generate text for a batch of prompts."""
        all_texts = []
        for prompt in prompts:
            sequences = self.pipeline(
                prompt,
                max_length=self.max_output_length,
            )
            # Extract generated text from the first sequence
            text = sequences[0]["generated_text"]
            logger.warning(f"Generated: {text}")
            all_texts.append(text)

        # Return batch output with shape [batch_size, -1]
        tensor = pb_utils.Tensor("text_output", np.array(all_texts, dtype=np.object_).reshape(-1, 1))
        output_tensors = [tensor]
        response = pb_utils.InferenceResponse(output_tensors=output_tensors)
        return response

    def finalize(self):
        print("Cleaning up...")