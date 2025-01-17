# Molecular Effectiveness Prediction App

## 🚀 Overview
This project is a **Molecular Effectiveness Prediction App** that predicts the **binding affinity (pIC50)** of a given molecule using a **Machine Learning model**. The app takes a **SMILES string** or a **molecular name** as input and returns the predicted pIC50 value, which helps in drug discovery or other therapeutic purposes by assessing how well a molecule binds to its biological target.

## 🎯 Features
- 🧪 Predicts **binding affinity (pIC50)** for a given molecular structure.
- 🖼️ Generates **2D molecular structure visualization**.
- 🔬 Converts **molecular names to SMILES** using Gemini AI.
- ⚡ **CI/CD pipeline** implemented for automated deployment.
- ✅ Database: PostgreSQL for data ingestion 🗄️
- 📊 **Prefect for pipeline orchestration**.
- 📈 **MLflow for experiment tracking and model management**.
- 🚀 **Deployed on Render** for real-time accessibility.

## 🏗️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python (FastAPI for API integration)
- **Database:** PostgreSQL 
- **Machine Learning:** RDKit, Scikit-learn, NumPy, XGBoost, etc
- **Orchestration:** Prefect
- **Experiment Tracking:** MLflow
- **Model Deployment:** Render (Dockerized container)
- **CI/CD:** GitHub Actions + Docker Hub

## 📦 Installation
### 1️⃣ Clone the Repository
```bash
https://github.com/Hindolch/Molecular_bioactivity_MLOps.git
cd molecular-prediction-app
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application Locally
```bash
streamlit run app/app.py --server.port 8000 --server.address 0.0.0.0
```

## 🐳 Docker Setup
### 1️⃣ Build the Docker Image
```bash
docker build -t chemical_image .
```

### 2️⃣ Run the Docker Container
```bash
docker run --name odntainer_name -p 8000:8000 chemical_image
```

## 🚀 Deployment on Render
This project is deployed on **Render** with automated updates using **CI/CD** pipelines. Any push to the main branch triggers a deployment.

## The deployed WebApp inference

![Screenshot from 2025-01-17 22-02-53](https://github.com/user-attachments/assets/798996cc-6e45-481e-ac31-d662f35bd761)

## The pipeline run

![Screenshot from 2025-01-17 22-59-33](https://github.com/user-attachments/assets/f9fbc819-4530-4a0f-a004-63ad7b07737d)

## MlFLow tracking 

![Screenshot from 2025-01-17 23-00-01](https://github.com/user-attachments/assets/fc533651-cfa6-42b8-b294-cc80a7b4f605)

## CI/CD deployment using GitHub Actions

![Screenshot from 2025-01-17 22-54-49](https://github.com/user-attachments/assets/5331899f-bd65-493c-94f5-ac2e1058397f)



## 🔬 Model Details
The prediction model is trained on molecular datasets using **fingerprint-based features** and classical machine learning models. Experiment tracking and versioning are handled by **MLflow**.

## 🛠️ CI/CD Pipeline
- **GitHub Actions** automates testing and deployment.
- **Docker Hub** stores the container image.
- **Render** pulls the latest Docker image and redeploys automatically.

## 📌 To-Do / Future Enhancements
- Add **3D molecular visualization** using Py3Dmol.
- Improve the model’s accuracy using **deep learning architectures**.
- Integrate **fastAPI backend** for a more scalable API.

## 🤝 Contribution
If you’d like to contribute, feel free to fork the repo and submit a pull request! 🚀

## 📜 License
This project is licensed under the MIT License.

## 📬 Contact
For any questions or collaborations, feel free to reach out!

🔗 **Deployed App:** [https://molecular-bioactivity-prediction.onrender.com/]

🔗 **LinkedIn:** [https://www.linkedin.com/in/hindol-choudhury-5ab5a5271/]
