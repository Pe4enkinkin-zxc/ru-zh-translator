from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate(texts, model_name="Helsinki-NLP/opus-mt-ru-zh", max_length=256, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    if device.startswith("cuda"):
        inputs = {k: v.cuda() for k,v in inputs.items()}

    outputs = model.generate(**inputs, max_length=max_length, num_beams=5)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
