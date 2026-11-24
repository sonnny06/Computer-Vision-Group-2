import time
from src.data.preprocessing import prepare_image
from src.utils.prompts import make_prompt

class MedicalVQAPipeline:
    def __init__(self,cfg,model=None):
        self.cfg=cfg
        self.model=model
    def _get_model(self):
        if self.model is None:
            from src.models.pretrained_vlm import get_vlm
            self.model=get_vlm(self.cfg)
        return self.model
    def predict(self,source,question="",task="vqa",answer_type="open",vision_context=None):
        c=self.cfg["inference"]
        if task not in ("vqa","caption","both"): raise ValueError("task: vqa/caption/both")
        if answer_type not in ("open","closed"): raise ValueError("answer_type: open/closed")
        prompt=make_prompt(question,"vqa" if task=="both" else task,c["language"],c["max_question_chars"])
        if answer_type=="closed" and task!="caption":
            prompt += " Trả lời chính xác một trong: Có; Không; Không đủ thông tin." if c["language"]=="vi" else " Answer exactly: Yes; No; Insufficient information."
        if vision_context:
            prompt += "\nExternal image analysis (unverified): " + vision_context
        image=prepare_image(source,c["max_image_side"],c["max_upload_mb"])
        start=time.perf_counter()
        text=self._get_model().generate(image,prompt,c["max_new_tokens"])
        from src.utils.postprocessing import format_answer
        caption = None
        if task=="both":
            caption_prompt=make_prompt("","caption",c["language"],c["max_question_chars"])
            if vision_context: caption_prompt += "\nExternal image analysis (unverified): " + vision_context
            caption=self._get_model().generate(image,caption_prompt,c["max_new_tokens"]).strip()
        raw=text
        text=format_answer(text,answer_type if task!="caption" else "open",c["language"])
        return {"answer":text,"raw_answer":raw,"caption":text if task=="caption" else caption,"answer_type":answer_type,
                "branches_executed":["vqa","caption"] if task=="both" else [task],"task":task,"model":self.cfg["model"]["id"],
                "language":c["language"],"elapsed_seconds":round(time.perf_counter()-start,3),
                "processed_image_size":list(image.size),"confidence":None,
                "note":"Educational prototype; medical accuracy not validated. No calibrated confidence available."}
