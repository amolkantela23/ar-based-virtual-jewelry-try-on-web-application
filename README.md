AR-BASED VIRTUAL JEWELRY TRY-ON WEB APPLICATION
================================================

An interactive web application that allows users to virtually try on jewelry (necklaces, earrings, nose pins) in real time using Augmented Reality (AR) and computer vision techniques.

------------------------------------------------
FEATURES
------------------------------------------------
- Real-time face, eye, and nose detection using OpenCV Haar Cascades
- Overlay of virtual jewelry (necklaces, earrings, nose pins) on detected facial features
- Built with HTML, CSS, and JavaScript for frontend
- Flask (Python) backend for real-time processing
- Works with webcam input for a live AR experience

------------------------------------------------
TECHNOLOGIES USED
------------------------------------------------
Frontend   : HTML, CSS, JavaScript
Backend    : Python (Flask)
Computer Vision : OpenCV
Other Tools: Haar Cascade Classifiers

------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------
AR-BASED-VIRTUAL-JEWELRY-TRY-ON-WEB-APPLICATION/
│── static/               # CSS, JS, and image assets
│── templates/            # HTML templates
│── haarcascades/         # Pre-trained Haar cascade XML files
│── app.py                # Flask backend
│── requirements.txt      # Dependencies
│── README.txt            # Project documentation

------------------------------------------------
INSTALLATION & SETUP
------------------------------------------------
1. Clone the repository:
   git clone https://github.com/your-username/ar-based-virtual-jewelry-try-on-web-application.git
   cd ar-based-virtual-jewelry-try-on-web-application

2. (Optional) Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate   (Windows)
   source venv/bin/activate (Mac/Linux)

3. Install dependencies:
   pip install -r requirements.txt

4. Run the application:
   python app.py

5. Open in browser:
   http://127.0.0.1:5000

------------------------------------------------
AUTHOR
------------------------------------------------
Developed by: Amol Kantela
