# GenderDetection Repository


Welcome to the **GenderDetection** repository! This repository features a gender identification AI software that classifies images of faces as male or female. Whether you are a fellow AI enthusiast, a student, or just curious about AI, you will find this project interesting and informative.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Installation
To get started with these projects, follow the installation instructions below:

```bash
git clone https://github.com/brendonmlopes/GenderDetection.git
cd GenderDetection
pip install -r requirements.txt
```
## Usage
As a default, the png images in the current directories are show in the terminal. You can provide the full path to the image, but it's easier to just put the images in the same folder as the faceDetect.py file.

Open your terminal and run the faceDetect.py file.
Provide the path of the image to be analyzed (like shown below)
```bash
python3 faceDetect.py 
Using device: cuda
im3.png
im5.png
im4.png
im1.png
im2.png
Path to your image:im1.png
```
Output:
```bash
The predicted gender is: Male
```
The gender_classifier.pth file is the file that holds the weights and biases for the network, so it MUST BE IN THE SAME DIRECTORY AS faceDetect.py. Changes may be done in the future, and the gender_classifier.pth can be updated for better accuracy. In that case, just download the new gender_classifier.pth and overwrite the old one.

## License
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contact
Email: brendonmaial@poli.ufrj.br
