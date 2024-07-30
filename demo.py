from PIL import Image
import torch
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import numpy as np
from torchvision import transforms
from models.clipseg import CLIPDensePredT
from googletrans import Translator
plt.style.use("dark_background")

@st.cache_data
def load_model():
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.load_state_dict(torch.load('E:\\clipseg\\weights\\rd64-uni.pth', map_location=torch.device('cpu')), strict=False)
    return model

def load_finetune():
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.load_state_dict(torch.load('E:/clipseg/weights_finetune/weights.pth', map_location=torch.device('cpu')), strict=False)
    return model

def rotate_image(image):
    image = image.convert("RGB")
    np_image = np.array(image)
    if np_image.shape[1] > np_image.shape[0]:
        np_image = cv2.rotate(np_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return Image.fromarray(np_image)

def process_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),  
    ])
    img = transform(image).unsqueeze(0)
    return img

def predict_segmentation(model, text_prompts, image):
    with torch.no_grad():
        image_tensor = torch.cat([image]*len(text_prompts), dim=0)
        preds = model(image_tensor, text_prompts)[0]
    
    masks = [torch.sigmoid(pred[0]) for pred in preds]
    masks = [(mask > 0.8).float().cpu().numpy() for mask in masks]
    return masks

def blend_image_with_mask(image, mask, alpha=0.5):
    image_np = np.array(image)
    mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))  
    mask_3ch = np.stack([mask_resized]*3, axis=-1)  
    blended = cv2.addWeighted(image_np, 1-alpha, (mask_3ch*255).astype(np.uint8), alpha, 0)
    return blended

def translate_text_prompts(text_prompts):
    translator = Translator()
    translated_prompts = []
    for prompt in text_prompts:
        try:
            translated_text = translator.translate(prompt, src='vi', dest='en').text
            if translated_text:
                translated_prompts.append(translated_text)
            else:
                translated_prompts.append(prompt)
        except Exception as e:
            translated_prompts.append(prompt)
            st.warning(f"Please input image and prompts !!")
    return translated_prompts

def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames

def frames_to_video(frames, output_path, fps):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()

def process_video(video_path, output_path, text_prompts, model, finetune_model):
    frames = extract_frames(video_path)
    processed_frames = []
    
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = process_image(image)
        masks = predict_segmentation(model, text_prompts, image_tensor)
        finetune_masks = predict_segmentation(finetune_model, text_prompts, image_tensor)
        
        for i, mask in enumerate(masks):
            blended_img = blend_image_with_mask(image, mask)
            processed_frames.append(cv2.cvtColor(np.array(blended_img), cv2.COLOR_RGB2BGR))
    
    frames_to_video(processed_frames, output_path, fps=30) 

def main():
    st.title("CLIPSeg Image/Video Segmentation with Text Prompts")
    st.subheader("By Nguyen Thanh Trung and Nguyen Hoang Quoc Huy")

    # file_type = st.radio("Choose file type:", ("Image", "Video"))
    
    text_prompts = st.text_input("Enter text prompts describing the objects to segment (separated by comma):")
    text_prompts = [prompt.strip() for prompt in text_prompts.split(',')]
    
    # if file_type == "Image":
    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
    # else:
    #     uploaded_file = st.file_uploader("Choose a video:", type=["mp4", "avi", "mov"])
    
    if text_prompts:
        text_prompts = translate_text_prompts(text_prompts)

    if uploaded_file is not None:
        if file_type == "Image":
            image = Image.open(uploaded_file)
            original_image = np.array(image)
            image_tensor = process_image(image)  

            if len(text_prompts) < 1:
                st.error("Please provide at least one text prompt.")
                return
            
            model = load_model()
            masks = predict_segmentation(model, text_prompts, image_tensor) 

            finetune_model = load_finetune()
            finetune_masks = predict_segmentation(finetune_model, text_prompts, image_tensor)  

            st.subheader("Segmentation with CLIPSeg model:")
            for i, mask in enumerate(masks):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(original_image, caption='Image', use_column_width=True)
                with col2:
                    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))  
                    st.image(mask_resized, caption=f'Segmentation Mask for {text_prompts[i]}', use_column_width=True)
                with col3:
                    blended_img = blend_image_with_mask(image, mask)
                    st.image(blended_img, caption=f'Blended Mask for {text_prompts[i]}', use_column_width=True)

            st.subheader("Segmentation with CLIPSeg finetuned model:")
            for i, mask in enumerate(finetune_masks):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(original_image, caption='Image', use_column_width=True)
                with col2:
                    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))  
                    st.image(mask_resized, caption=f'Segmentation Mask for {text_prompts[i]}', use_column_width=True)
                with col3:
                    blended_img = blend_image_with_mask(image, mask)
                    st.image(blended_img, caption=f'Blended Mask for {text_prompts[i]}', use_column_width=True)

        elif file_type == "Video":
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())

            model = load_model()
            finetune_model = load_finetune()
            process_video("temp_video.mp4", "output_video.mp4", text_prompts, model, finetune_model)
            st.video("output_video.mp4")


if __name__ == "__main__":
    main()
