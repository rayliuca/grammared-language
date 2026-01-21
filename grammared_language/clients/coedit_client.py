from .text2text_base_client import Text2TextBaseClient
from typing import Optional


class CoEditClient(Text2TextBaseClient):
    """
    Client for CoEdit grammar correction models on Triton Inference Server.
    
    CoEdit is a text editing model trained for grammar correction and text refinement.
    This client provides a convenient interface with sensible defaults for CoEdit models.
    
    Model: grammarly/coedit-large, grammarly/coedit-xl
    Paper: https://arxiv.org/abs/2305.09857
    
    Args:
        model_name: Name of the Triton model (default: "coedit_large")
        triton_host: Triton server host (default: "localhost")
        triton_port: Triton server port (default: 8000)
        triton_model_version: Model version (default: "1")
        task: Task type for instruction template. Options:
            - "grammar": Fix grammatical errors (default)
            - "fluency": Make the text more fluent
            - "coherence": Make the text more coherent
            - "clarity": Make the text more clear
            - "paraphrase": Paraphrase the text
            - "neutralize": Make the text more neutral
            - "simplify": Simplify the text
            - "formalize": Make the text more formal
            - "update": Update information in the text
            - None: No task prefix (model receives raw text)
        chat_template: Custom chat template to override task-based templates.
                      Use {text} as placeholder. Example: "Fix grammar: {text}"
                      For chat-style: "<|user|>\n{text}<|assistant|>\n"
        **kwargs: Additional arguments passed to Text2TextBaseClient
    
    Example:
        >>> client = CoEditClient(task="grammar")
        >>> result = client.predict("She go to the store yesterday.")
        >>> print(result.matches)
    """
    
    # CoEdit task prefixes based on the paper
    TASK_PROMPTS = {
        "grammar": "Fix grammatical errors: {text}",
        "fluency": "Make the text more fluent: {text}",
        "coherence": "Make the text more coherent: {text}",
        "clarity": "Make the text more clear: {text}",
        "paraphrase": "Paraphrase: {text}",
        "neutralize": "Make the text more neutral: {text}",
        "simplify": "Simplify: {text}",
        "formalize": "Make the text more formal: {text}",
        "update": "Update: {text}",
    }
    
    def __init__(
        self,
        model_name: str = "coedit_large",
        *,
        triton_host: str = "localhost",
        triton_port: int = 8000,
        triton_model_version: str = "1",
        task: Optional[str] = "grammar",
        chat_template: Optional[str] = None,
        **kwargs
    ):
        # Determine chat template
        if chat_template is not None:
            template = chat_template
        elif task is not None:
            if task not in self.TASK_PROMPTS:
                raise ValueError(
                    f"Invalid task: {task}. Must be one of: {list(self.TASK_PROMPTS.keys())} or None"
                )
            template = self.TASK_PROMPTS[task]
        else:
            template = None
        
        # Initialize parent with CoEdit-specific defaults
        super().__init__(
            model_name=model_name,
            triton_host=triton_host,
            triton_port=triton_port,
            triton_model_version=triton_model_version,
            input_name="text_input",
            output_name="text_output",
            chat_template=template,
            **kwargs
        )
        
        self.task = task
