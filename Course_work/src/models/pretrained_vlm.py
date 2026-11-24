import threading
import time

import torch
from safetensors import SafetensorError

from src.data.preprocessing import prepare_image


class VLMError(RuntimeError):
    """The pretrained model could not be loaded or run."""


class VLMOutOfMemoryError(VLMError):
    """CUDA ran out of memory during loading or inference."""


class PretrainedVLM:
    def __init__(self, cfg):
        settings = cfg["model"]
        self.model_id = settings["id"]
        self.backend = settings["backend"]

        device = settings["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            if torch.version.cuda is None or not torch.cuda.is_available():
                raise VLMError(
                    "CUDA NVIDIA không khả dụng. Kiểm tra nvidia-smi, driver "
                    "và bản PyTorch có CUDA."
                )
            device = "cuda:0"
        self.device = device
        self.cuda = device.startswith("cuda")
        self.lock = threading.Lock()

        dtype = settings.get("dtype", "auto")
        if dtype == "auto":
            if not self.cuda:
                dtype = "float32"
            else:
                with torch.cuda.device(self.device):
                    dtype = (
                        "bfloat16"
                        if torch.cuda.is_bf16_supported(including_emulation=False)
                        else "float16"
                    )
        self.dtype = getattr(torch, dtype)

        quantized = settings["load_in_4bit"]
        if quantized and not self.cuda:
            raise ValueError("Chế độ 4-bit yêu cầu CUDA NVIDIA.")

        local_only = settings["local_files_only"]
        try:
            self.model, self.processor = self._load(local_only, quantized)
        except torch.cuda.OutOfMemoryError:
            pass
        except (OSError, RuntimeError, ValueError, ImportError, SafetensorError) as exc:
            if local_only:
                hint = "Kiểm tra cache hoặc đặt local_files_only: false để tải đủ tệp."
            else:
                hint = "Kiểm tra kết nối Hugging Face, dung lượng cache và thư viện."
            if quantized:
                hint += " Kiểm tra bitsandbytes trong requirements-4bit.txt."
            raise VLMError(
                f"Không tải được {self.model_id} trên {self.device}. {hint} {exc}"
            ) from exc
        else:
            return

        hint = "Giải phóng VRAM rồi thử lại."
        if not quantized:
            hint += " Có thể cài requirements-4bit.txt và bật load_in_4bit."
        self._raise_cuda_oom("khi nạp model", hint)

    def _load(self, local_files_only, quantized):
        from transformers import AutoProcessor

        kwargs = {
            "local_files_only": local_files_only,
            "dtype": self.dtype,
            "attn_implementation": "sdpa",
        }
        if self.cuda:
            kwargs["device_map"] = {"": self.device}
        if quantized:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )
        if self.backend == "qwen":
            from transformers import Qwen3VLForConditionalGeneration as Model
        else:
            from transformers import AutoModelForImageTextToText as Model

        processor = AutoProcessor.from_pretrained(
            self.model_id, local_files_only=local_files_only
        )
        model = Model.from_pretrained(self.model_id, **kwargs).eval()
        return model, processor

    def _raise_cuda_oom(self, stage, hint):
        torch.cuda.empty_cache()
        raise VLMOutOfMemoryError(f"Hết VRAM {stage} ({self.device}). {hint}")

    def generate(self, image, prompt, max_new_tokens):
        """Return text for a PIL image already prepared by the pipeline."""
        return self._run(image, prompt, max_new_tokens, measure=False)

    def infer(
        self, image, prompt, max_new_tokens=128, *,
        max_image_side=768, max_upload_mb=15,
    ):
        """Accept an image source and prompt. Return text and metrics."""
        image = prepare_image(image, max_image_side, max_upload_mb)
        return self._run(image, prompt, max_new_tokens, measure=True)

    def _run(self, image, prompt, max_new_tokens, *, measure):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt phải là chuỗi có nội dung.")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens phải là số nguyên dương.")

        with self.lock:
            measure_cuda = measure and self.device.startswith("cuda")
            if measure_cuda:
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                baseline = torch.cuda.memory_allocated(self.device)
            if measure:
                start = time.perf_counter()

            try:
                answer, tokens = self._generate(image, prompt, max_new_tokens)
                if measure_cuda:
                    torch.cuda.synchronize(self.device)
            except torch.cuda.OutOfMemoryError:
                pass
            else:
                if not measure:
                    return answer
                metrics = {
                    "device": self.device,
                    "dtype": str(self.dtype),
                    "inference_seconds": time.perf_counter() - start,
                    "generated_tokens": tokens,
                    "cuda_memory": None,
                }
                if measure_cuda:
                    metrics["cuda_memory"] = {
                        "baseline_allocated_bytes": baseline,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                    }
                return {"answer": answer, "metrics": metrics}

            self._raise_cuda_oom(
                "khi suy luận", "Giảm max_image_side hoặc max_new_tokens rồi thử lại."
            )

    @torch.inference_mode()
    def _generate(self, image, prompt, max_new_tokens):
        image_content = {"type": "image"}
        messages = [{
            "role": "user",
            "content": [image_content, {"type": "text", "text": prompt}],
        }]
        if self.backend == "qwen":
            image_content["image"] = image
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[image], return_tensors="pt")

        inputs = {
            key: value.to(device=self.device, dtype=self.dtype)
            if value.is_floating_point() else value.to(self.device)
            for key, value in inputs.items()
        }
        ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
        generated = ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if not answer:
            raise VLMError("Model trả về câu trả lời rỗng.")
        return answer, generated.shape[1]


_models = {}
_models_lock = threading.Lock()


def get_vlm(cfg=None):
    """Load once per model configuration in this Python process."""
    if cfg is None:
        from src.utils.config import ROOT, load_config

        cfg = load_config(ROOT / "configs/model_config.yaml")

    settings = dict(cfg["model"])
    if settings["device"] == "auto":
        settings["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    key = (
        settings["backend"],
        settings["id"],
        settings["device"],
        settings.get("dtype", "auto"),
        settings["load_in_4bit"],
        settings["local_files_only"],
    )
    with _models_lock:
        if key not in _models:
            _models[key] = PretrainedVLM({"model": settings})
        return _models[key]