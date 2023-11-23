# ModaMate - Pose Landmark Detection

ModaMate is an innovative Augmented Reality (AR) Virtual Try-On Application designed to enhance the shopping experience by allowing users to virtually try on various outfits and accessories. The `moda_model_landmarks.py` script is a crucial component of this app, focusing on extracting 2D pose landmarks using MediaPipe's holistic solutions.

## Features
- **Webcam Support**: Captures video from a webcam in real-time and extracts 2D pose landmarks.
- **Image Processing**: Allows for processing of static images to extract pose landmarks.
- **Landmark Visualization**: Draws landmarks on both live webcam footage and static images for immediate visual feedback.
- **Custom Output**: Generates images with landmarks drawn on a black background, ideal for further processing or demonstration purposes.

## Importance in ModaMate
- **Pose Estimation**: Accurate pose estimation is fundamental for the virtual try-on feature, ensuring that outfits and accessories align correctly with the user's body in real-time.
- **User Engagement**: Enhances user experience by providing a realistic and interactive way of trying on different styles virtually.
- **Development Flexibility**: Offers a modular approach, allowing easy integration with other components of the ModaMate app.

### *Note*: Its advisable to run this script within a python virtual environment
```bash
virtualenv moda_mate_env
source moda_mate_env/bin/activate 
```

## Getting Started
### Prerequisites
- Python 3.x
- OpenCV
- MediaPipe


### Installation
1. Clone the repository:
```bash
    git clone [repository-url]
```
2. Navigate to the script's directory:
```bash
    cd [script-directory]
```
3. Install dependencies:
```bash
    pip install -r requirements.txt
```

### Usage
Run the script using Python:
```
python moda_model_landmark.py
```
Follow the prompts to choose between capturing from webcam or processing a static image.

## Contributing
Contributions to ModaMate are welcome. Please ensure to follow the contributing guidelines and code of conduct.

## License


## Acknowledgments
- Special thanks to the MediaPipe team for providing powerful tools for pose estimation.
- Contributors and supporters of the ModaMate project.