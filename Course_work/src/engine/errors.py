class PipelineError(Exception):
    code = "PIPELINE_ERROR"


class InvalidImageError(PipelineError):
    code = "INVALID_IMAGE"


class InvalidQuestionError(PipelineError):
    code = "INVALID_QUESTION"


class PromptError(PipelineError):
    code = "PROMPT_ERROR"


class ModelUnavailableError(PipelineError):
    code = "MODEL_UNAVAILABLE"


class InferenceError(PipelineError):
    code = "INFERENCE_FAILED"


class InferenceOutOfMemory(PipelineError):
    code = "OUT_OF_MEMORY"


class InferenceTimeout(PipelineError):
    code = "INFERENCE_TIMEOUT"