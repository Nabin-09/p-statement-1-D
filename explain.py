from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layer = model.cnn.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

grayscale_cam = cam(input_tensor=img)[0]
visualization = show_cam_on_image(img.cpu().numpy()[0].transpose(1,2,0), grayscale_cam)

cv2.imwrite("explanation.jpg", visualization)
