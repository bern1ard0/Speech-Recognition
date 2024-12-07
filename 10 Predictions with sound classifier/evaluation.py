import torch
from torch.utils.data import DataLoader
from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset
import torchaudio

# Parameters
MODEL_PATH = "cnnnet.pth"
ANNOTATIONS_FILE = "/Users/bernardwongibe/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "/Users/bernardwongibe/Downloads/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 128

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = CNNNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# Prepare dataset and dataloader
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

dataset = UrbanSoundDataset(
    ANNOTATIONS_FILE,
    AUDIO_DIR,
    mel_spectrogram,
    SAMPLE_RATE,
    NUM_SAMPLES,
    device
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Evaluation function
def evaluate(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

# Run evaluation
evaluate(model, dataloader)