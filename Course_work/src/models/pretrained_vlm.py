import threading
import torch

class PretrainedVLM:
    """One pretrained VLM shared by VQA and captioning; no training or custom heads."""
    def __init__(self, cfg):
        from transformers import AutoProcessor, AutoModelForVision2Seq, Qwen3VLForConditionalGeneration
        m=cfg["model"]
        self.backend=m["backend"]
        self.device=("cuda" if torch.cuda.is_available() else "cpu") if m["device"]=="auto" else m["device"]
        if self.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA không khả dụng")
        self.dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device=="cuda" else torch.float32
        kw={"local_files_only":m["local_files_only"],"torch_dtype":self.dtype,"attn_implementation":"sdpa"}
        if m["load_in_4bit"]:
            if self.device!="cuda": raise ValueError("4-bit profile requires NVIDIA CUDA; disable on CPU")
            from transformers import BitsAndBytesConfig
            kw["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=self.dtype,bnb_4bit_use_double_quant=True)
            kw["device_map"]={"":"cuda:0"}
        self.processor=AutoProcessor.from_pretrained(m["id"],local_files_only=m["local_files_only"])
        cls=Qwen3VLForConditionalGeneration if self.backend=="qwen" else AutoModelForVision2Seq
        self.model=cls.from_pretrained(m["id"],**kw)
        if not m["load_in_4bit"]: self.model.to(self.device)
        self.model.eval()
        self.lock=threading.Lock()
    @torch.inference_mode()
    def generate(self,image,prompt,max_new_tokens):
        with self.lock:
            messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
            if self.backend=="qwen":
                messages[0]["content"][0]["image"]=image
                inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")
            else:
                text=self.processor.apply_chat_template(messages,add_generation_prompt=True)
                inputs=self.processor(text=text,images=[image],return_tensors="pt")
            # Move image floats to the model dtype; keep token IDs and masks integer.
            inputs={k:(v.to(device=self.device,dtype=self.dtype) if v.is_floating_point() else v.to(self.device)) for k,v in inputs.items()}
            ids=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True)
            generated=ids[:,inputs["input_ids"].shape[1]:]
            answer=self.processor.batch_decode(generated,skip_special_tokens=True)[0].strip()
            if not answer: raise RuntimeError("Model returned an empty answer")
            return answer
