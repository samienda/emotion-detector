# Emotion Detection System

This project is an emotion detection system that uses a pre-trained convolutional neural network to detect emotions from facial images in real-time.

## Setup Instructions

Follow these steps to set up and run the project on your local machine:

### 1. Create a Virtual Environment

Open a terminal or command prompt, navigate to the project directory, and create a virtual environment:

```sh
python -m venv venv
```

### 2. Activate the virtual environment:

On Windows:
```sh
venv\Scripts\activate
```

On macOS and Linux:
```sh
source venv/bin/activate
```

### 3. Install Dependencies

With the virtual environment activated, install the project dependencies by running:

```sh
pip install -r requirements.txt
```

### 4. Run the Application

To start the emotion detection system, execute:

```sh
python detect_emotions.py
```

Make sure you have a webcam connected for the application to function properly.

## Usage

Once the application is running, it will automatically use your webcam to detect faces and predict their emotions in real-time. The interface will display the detected emotion on the screen.