import requests
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


def init_model(model_name, device=torch.device("cuda")):
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device
    ).eval()

    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
    model.processor = processor

    return model


def get_response_concat(model, question, image_path_list, max_new_tokens=1024, temperature=1.0):
    images = []
    for img_path in image_path_list:
        if img_path.startswith(('http://', 'https://')):
            image = Image.open(requests.get(img_path, stream=True).raw)
        else:
            image = Image.open(img_path)
        images.append(image)

    conversation = [
        {
            "role": "user",
            "content": [
                           {"type": "text", "text": question},
                       ] + [{"type": "image"} for _ in range(len(images))]
        },
    ]

    prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = model.processor(images=images, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = model.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]
