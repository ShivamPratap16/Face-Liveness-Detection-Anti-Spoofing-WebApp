# 🎭 Face Liveness Detection (Anti-Spoofing) Web App

**Defend your system from impostors!** This web app smartly discerns whether a face belongs to a living person or a mere replica, ensuring your verifications are always genuine and secure.

---

## 🚀 Why We Built This

In a world where digital security is paramount, simple face recognition isn't enough. Inspired by [this awesome project](https://github.com/jomariya23156/face-recognition-with-liveness-web-login), we've taken it a step further with advanced liveness detection to prevent spoofing attacks.

---

## 🛠 Proposed Solution

### Strengthening Aadhaar Authentication with Face Liveness

A biometric authentication system using **TensorFlow.js** and **ONNX.js** for live user detection. The solution uses TensorFlow.js for client-side biometric authentication, detecting live users to prevent spoofing. It integrates with Aadhaar and scales across devices, combining passive and active liveness detection. Runs in browsers with detection under 500ms.

---

## 🧩 Problem Addressed

- **Prevents Spoofing**: Detects live users versus photos, videos, or masks.
- **User-Friendly**: Simpler than fingerprints or iris scans, catering to accessibility needs.
- **Seamless Remote Authentication**: Enables secure Aadhaar-based remote authentication.
- **Cross-Domain Adaptability**: Addresses challenges in diverse conditions and attack types with robust models and datasets.

---

## 💡 Innovation

- **Hybrid Detection**: Combines active and passive methods for stronger spoof resistance.
- **Inclusive Design**: Uses diverse datasets for accurate liveness detection across India's demographics.

---

## 🔍 Feasibility and Viability

### Feasibility Analysis:
- **Optimized for High-Performance Inference**: Utilizes TensorFlow.js for real-time inference on low-latency applications, supporting GPU acceleration and WebAssembly for enhanced performance on the client side.
- **Seamless Integration**: Integrates with Aadhaar and other large-scale identity systems, ensuring secure, compliant, and efficient authentication for widespread applications.

### Potential Challenges and Risks:
- Ensuring accuracy in varied environments and lighting.
- Performance issues on low-end devices and compatibility with different browsers.
- Addressing demographic diversity in India.
- Adoption resistance in low digital literacy areas.

### Strategies for Overcoming Challenges:
- Regular model updates for accuracy under all kinds of environments.
- Optimization for low-end devices and usage of lightweight models for compatibility across all browsers.
- Focus on updating real-time datasets.
- Easy-to-understand and user-friendly interface.

---

## 🌟 Impact and Benefits

### Potential Impact:
- Reduces the risk of fraudulent activities like face spoofing.
- Enhances accuracy through vast amounts of real-time training data.
- Wider adoption in digitally underserved communities.
- Provides a robust model, supporting large volumes of authentication requests for a big population.
- Future-proof model, reliable for long-term use in the ever-changing landscape of the cyber world.

### Benefits:
- **Social**: Boosts security and accessibility, aids those with disabilities, fosters trust among users and stakeholders, and enhances accuracy across various skin tones, facial structures, and lighting.
- **Economic**: Cuts costs via client-side processing and less infrastructure, along with low maintenance costs compared to hardware solutions.
- **Environmental**: Decreases energy use by minimizing server reliance.

---

## 📸 Screenshots

### Website Prototype
![Webcam Page](static/assets/img/image2.png)
### Technical Approach
![Face Liveness Detector](static/assets/img/image.png)

---
## ⚡ Quick Start

Get the project running locally in just a few steps:

### Step 1: Clone the Repository  
Fork and clone the repo locally:

```sh
git clone https://github.com/ShivamPratap16/Face-Liveness-Detection-Anti-Spoofing-WebApp.git
```

### Step 2: Create and Activate a Virtual Environment  
To keep dependencies organized, create and activate a virtual environment:

```sh
pip install virtualenv
python -m venv [env-name]
source [env-name]/bin/activate  # For MacOS/Linux
[env-name]\Scripts\activate     # For Windows
```

### Step 3: Navigate to Project Directory  

```sh
cd Face-Liveness-Detection-Anti-Spoofing-Web-App
```

### Step 4: Install Dependencies  

```sh
pip install -r requirements.txt
```

### Step 5: Run the App  

```sh
streamlit run app.py
```
This will launch the Streamlit app in your default web browser.

🤝 Contributing
Contributions are welcome! Feel free to fork the repository, make enhancements, and open a pull request.#
