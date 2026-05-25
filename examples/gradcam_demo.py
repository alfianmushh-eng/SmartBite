"""Grad-CAM heatmap visualization example."""

import torch
import numpy as np
from smartbite.models.classifier import FoodClassifier
from smartbite.visualization.heatmap import GradCAM, generate_heatmap


def main():
    model = FoodClassifier(num_classes=210, pretrained=False)
    target_layer = model.backbone.features[-1]

    cam = GradCAM(model, target_layer)
    x = torch.randn(1, 3, 224, 224)
    heatmap = cam.generate(x)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    dummy_img = np.uint8(255 * np.random.random((224, 224, 3)))
    overlay = generate_heatmap(dummy_img, heatmap)
    print(f"Overlay image shape: {overlay.shape}")
    print("Grad-CAM visualisation ready. Save output with cv2.imwrite().")


if __name__ == "__main__":
    main()
