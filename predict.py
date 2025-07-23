import torch
from torchvision import models, transforms
from PIL import Image
import sys

def predict_the_class(image_path, model_path='model.pth'):
    """
    Funkcija na osnovu fotografije i modela ocijeni koja je klasa originalnog lavirinta
    :param image_path: putanja do fotografije
    :param model_path: istrenirani model
    :return: broj klase od 0 do 5, gje 0 znaci slika1.png, a 5 znači slika6.png
    """
    # Transformacija kao kod treniranja
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Učitavanje slike
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Učitavanje modela
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Predikcija
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

def main():
    # Putanja do slike i modela
    image_path = sys.argv[1]  # npr. python predict.py lavirint1.jpg
    predicted_class = predict_the_class(image_path)
    print(f"Predviđena klasa: {predicted_class} (slika{predicted_class + 1}.png)")


if __name__ == "__main__":
    main()