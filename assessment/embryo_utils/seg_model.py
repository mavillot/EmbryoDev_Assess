import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


transform_image = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

def load_model(checkpoint, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    torch._C._jit_set_profiling_mode(False)
    model = torch.jit.load(checkpoint)
    model = model.to(device)
    model.eval()
    return model



def predict_mask(img: np.ndarray, checkpoint: str, task:str ,target_size: tuple = (480, 480)) -> np.ndarray:
        """Predict segmentation mask for a single image."""
        # Load and preprocess image
        image=Image.fromarray(img)
        orig_w, orig_h = image.size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Resize for model
        image = transforms.Resize(target_size)(image)
        input_tensor = transform_image(image).unsqueeze(0).to(device)
        
        #load model
        model = load_model(checkpoint, device=device)
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Post-process
        outputs = torch.argmax(outputs, 1)
        mask = outputs.squeeze(0).cpu().numpy()

        if task == 'Fragmentation':
            mask[mask == 1] = 255
        elif task == 'Segmentation':
            mask[mask == 1] = 75
            mask[mask == 2] = 255
            mask[mask == 3] = 150
        
        # Resize back to original size
        mask_resized = np.array(
            transforms.Resize((orig_h, orig_w))(Image.fromarray(mask.astype('uint8')))
        )
        return mask_resized

