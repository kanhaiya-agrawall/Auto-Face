# Auto-Face: Automatic Facial Recognition-Based Attendance System

## Overview

Auto-Face is an automatic facial recognition-based attendance system that utilizes computer vision techniques to detect and identify faces. This system records attendance by recognizing individuals from live camera feeds and logging their presence into an attendance report. The project leverages OpenCV and NumPy for face detection and K-Nearest Neighbors (KNN) for face recognition.

## Features

- **Real-time Face Detection:** Uses Haar cascades to detect faces in real-time.
- **Face Recognition:** Employs KNN algorithm to recognize and classify faces.
- **Attendance Logging:** Automatically logs recognized individuals' attendance with timestamp.
- **Data Storage:** Stores face data as NumPy arrays for training the recognition model.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pandas

You can install the required Python packages using pip:

```bash
pip install opencv-python-headless numpy pandas
```
## Project Structure

- `face_data/` - Directory to store face data files.
- `Detection_And_Storage_Open_CV.py` - Script to collect and store face data.
- `KNN_Model_Attendance_Recorder.py` - Script to perform face recognition and log attendance.
- `haarcascade_frontalface_default.xml` - Pre-trained Haar cascade classifier for face detection.

## Usage

### Collect Face Data

Run the `Detection_And_Storage_Open_CV.py` script to collect and store face data for individuals. This script will capture frames from the camera, detect faces, and save face data in the `face_data/` directory.

```bash
python Detection_And_Storage_Open_CV.py
```
### Record Attendance

Run the `KNN_Model_Attendance_Recorder.py` script to recognize faces and record attendance. This script will continuously capture frames from the camera, recognize faces, and log attendance in a CSV file.

```bash
KNN_Model_Attendance_Recorder.py
```

## Notes

- Ensure the haarcascade_frontalface_default.xml file is in the same directory as the scripts or provide the correct path in the scripts.
- The attendance report CSV file will be created in the same directory as the script.

## License
This project is licensed under the MIT License.

### Acknowledgments

- OpenCV for computer vision techniques.
- NumPy for numerical operations.
- Pandas for handling CSV file operations.






