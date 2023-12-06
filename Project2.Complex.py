import os
import tkinter as tk
from tkinter import ttk, filedialog
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image, ImageTk
import numpy as np

# Define a simple neural network model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# Initialize global variables
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = None  # Define train_loader as a global variable

# Function to preprocess the image before feeding it to the neural network
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    preprocessed_image = transform(image).unsqueeze(0)
    return preprocessed_image

# Function to train the neural network with additional metrics
def train_neural_network_with_metrics(model, train_loader, criterion, optimizer, num_epochs=30):
    if train_loader is None:
        print("Please select a folder to load the dataset.")
        return

    for epoch in range(num_epochs):
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

        # Print metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            average_loss = running_loss / len(train_loader)
            accuracy = accuracy_score(all_labels, all_predictions)
            confusion_mat = confusion_matrix(all_labels, all_predictions)

            print(f'Epoch {epoch + 1}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2%}')
            print('Confusion Matrix:')
            print(confusion_mat)
            print('-' * 50)


# Function to make a prediction using the neural network
def make_prediction(image_tensor, model):
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Function to handle the "Load Image" button
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300))
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo
        prediction_label.config(text="Prediction: ")

        # Preprocess the image and make a prediction
        preprocessed_image = preprocess_image(file_path)
        predicted_class = make_prediction(preprocessed_image, model)
        prediction_label.config(text=f"Prediction: Class {predicted_class}")

# Function to handle the "Clear Image" button
def clear_image():
    image_label.config(image="")
    image_label.image = None
    prediction_label.config(text="Prediction: ")

# Function to load the MNIST dataset from the specified folder
def load_mnist_dataset(folder_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_dataset = datasets.ImageFolder(root=folder_path, transform=transform)
    mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=64, shuffle=True)

    return mnist_loader

# Function to handle the "Select Folder" button
def select_folder():
    global model  # Access the global model variable
    folder_path = filedialog.askdirectory()
    if folder_path:
        train_loader = load_mnist_dataset(folder_path)

        # Reinitialize the neural network, criterion, and optimizer
        model = CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Retrain the neural network with metrics (replace this with your actual training loop)
        train_neural_network_with_metrics(model, train_loader, criterion, optimizer, num_epochs=10)

# Function to handle the "Train another 10 epochs" button
def train_additional_epochs():
    global model, train_loader, criterion, optimizer
    # Train another 10 epochs
    train_neural_network_with_metrics(model, train_loader, criterion, optimizer, num_epochs=10)

# Create the main window
root = tk.Tk()
root.title("MNIST GUESSER")
root.configure(bg='#37474F')  # Set background color

# Create and place widgets with a modern aesthetic
title_label = ttk.Label(root, text="Neural Network Image Guesser", font=("Helvetica", 16), foreground='#ECEFF1', background='#37474F')
title_label.pack(pady=10)

image_label = ttk.Label(root, background='#263238')
image_label.pack(pady=10)

load_button = ttk.Button(root, text="Load Image", command=load_image, style='DarkGray.TButton')
load_button.pack()

clear_button = ttk.Button(root, text="Clear Image", command=clear_image, style='DarkGray.TButton')
clear_button.pack(pady=10)

select_folder_button = ttk.Button(root, text="Select Folder", command=select_folder, style='DarkGray.TButton')
select_folder_button.pack(pady=10)

train_additional_button = ttk.Button(root, text="Train another 10 epoch", command=train_additional_epochs, style='DarkGray.TButton')
train_additional_button.pack(pady=10)

prediction_label = ttk.Label(root, text="Prediction: ", font=("Helvetica", 12), foreground='#ECEFF1', background='#37474F')
prediction_label.pack(pady=10)

# Define a custom style for dark gray buttons with white text
style = ttk.Style()
style.configure('DarkGray.TButton', foreground='#424242', background='#424242', font=("Helvetica", 12))

# Start the Tkinter event loop
root.mainloop()