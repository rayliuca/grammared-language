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


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)  # Set default level to WARNING



DEFAULT_MODEL_BACKEND = "transformers"
def load_pipeline_from_config(model_config, task="text2text-generation", backend:str="transformers", fallback:bool=True, **kwargs):
    """Load a HuggingFace model based on the provided configuration.

    Args:
        model_config (dict): Model configuration dictionary.

    Returns:
        model: Loaded HuggingFace model.
    """
    # Extract model parameters from configuration
    model_params = model_config.get("parameters", {})
    default_hf_model = "HuggingFaceTB/SmolLM2-135M"
    if 'pretrained_model_name_or_path' in model_config['parameters']:
        hf_model = model_config['parameters']['pretrained_model_name_or_path']['string_value']
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
        self.grammared_language_model_config = json.loads(self.model_config.get('grammared_language_model_config', "{}"))
        default_max_gen_length = "15"
        # Check for user-specified model name in model config parameters
        hf_model = self.model_params.get("pretrained_model_name_or_path", {}).get(
            "string_value", self.DEFAULT_HF_MODEL
        )
        # Check for user-specified max length in model config parameters
        self.max_output_length = int(
            self.model_params.get("max_output_length", {}).get(
                "string_value", default_max_gen_length
            )
        )

        self.logger.log_info(f"Max output length: {self.max_output_length}")
        self.logger.log_info(f"Loading HuggingFace model: {hf_model}...")

        # Get model instance device configuration
        model_device_type = args['model_instance_kind']
        model_instance_device_id = args['model_instance_device_id']
        print("Model instance kind:", model_device_type)
        print("Model instance device id:", model_instance_device_id)
        # Determine device
        if model_device_type == 'CPU':
            self.device = 'cpu'
        else:
            self.device = f'cuda:{model_instance_device_id}'

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
            self.model_config,
            task=self.PIPELINE_TASK,
            backend=self.model_params.get("model_backend", {}).get("string_value", DEFAULT_MODEL_BACKEND),
            fallback=True,
            device=self.device,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            # Assume input named "prompt", specified in autocomplete above
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            prompt = input_tensor.as_numpy()[0].decode("utf-8")

            self.logger.log_info(f"Generating sequences for text_input: {prompt}")
            response = self.generate(prompt)
            responses.append(response)

        return responses

    def generate(self, prompt):
        sequences = self.pipeline(
            prompt,
            max_length=self.max_output_length,
        )

        output_tensors = []
        texts = []
        for i, seq in enumerate(sequences):
            text = seq["generated_text"]
            self.logger.log_info(f"Sequence {i+1}: {text}")
            texts.append(text)

        tensor = pb_utils.Tensor("text_output", np.array(texts, dtype=np.object_))
        output_tensors.append(tensor)
        response = pb_utils.InferenceResponse(output_tensors=output_tensors)
        return response

    def finalize(self):
        print("Cleaning up...")