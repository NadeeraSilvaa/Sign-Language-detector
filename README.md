# Sign Language Detector

This project is a sign language detector that uses a Python-based system to create a dataset using a webcam, train a model on that data, and test the model. It leverages OpenCV for video capture, MediaPipe for hand tracking, and scikit-learn for model training and evaluation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Dependencies](#dependencies)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/sign-language-detector.git
    cd sign-language-detector
    ```

2. **Create a virtual environment and activate it:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Collect data using the webcam:**

    Run `collect_data.py` to start collecting sign language data. Follow the prompts to capture images for different signs.

    ```sh
    python collect_data.py
    ```

2. **Create the dataset:**

    Use `create_dataset.py` to process the collected data and prepare it for training.

    ```sh
    python create_dataset.py
    ```

3. **Train the classifier:**

    Train your model using `train_classifier.py`.

    ```sh
    python train_classifier.py
    ```

4. **Test the model:**

    Evaluate the performance of your trained model using `test_model.py`.

    ```sh
    python test_model.py
    ```

## File Descriptions

- `collect_data.py`: Script to collect sign language data using the webcam.
- `create_dataset.py`: Processes the collected data and prepares it for training.
- `train_classifier.py`: Trains a machine learning model on the processed dataset.
- `test_model.py`: Tests the trained model and evaluates its performance.
- `requirements.txt`: Lists the dependencies required for this project.

## Dependencies

This project uses the following Python libraries:

- `opencv-python`: For video capture and image processing.
- `mediapipe`: For hand tracking and landmark detection.
- `scikit-learn`: For building and evaluating the machine learning model.

To install these dependencies, run:

```sh
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
