## 🌿 ** Project:**
**Plant Disease Detection Web App** that can:
- Take an image of a plant leaf 📷
- Predict which disease (if any) the plant has 🦠
- Show the result with confidence and a visual 🖼️

Project done using:
- A **deep learning model** trained on leaf images (with TensorFlow)
- A **Flask web app** so users can upload an image and get instant results in a browser

---

## 🧠 **How It Works**

---

### ✅ Step 1: You trained a model (`model-training.py`)
- You started with a **ZIP file** that had lots of leaf images (healthy + diseased).
- You **unpacked** it and prepared it for training.
- You used a **Convolutional Neural Network (CNN)** to learn patterns from those images.
- The model looked at:
  - Color
  - Shape
  - Texture
  - Spots or blights
- You trained the model and **saved** it as `plant_disease_model.h5`.
- You also saved the list of diseases (`class_indices.json`) so the model knows which number means which disease.

---

### ✅ Step 2: You created a web app (`app.py`)
- When the app runs, it:
  1. **Loads your trained model**
  2. **Shows a simple website**
  3. Lets the user **upload a leaf photo**
  4. **Processes** the photo (resizes it to 224x224 pixels)
  5. Sends it to the model for prediction
  6. Gets the result — like:  
     ➤ “Tomato - Late Blight (Confidence: 96%)”
  7. Creates an image with the prediction written on it
  8. Sends that back to the user to **see the result directly in the browser**

---

## 🔧 Tools and Technologies You Used:

| Technology | Role |
|------------|------|
| **TensorFlow/Keras** | For training the image recognition model |
| **Flask** | For building the web interface (backend + routing) |
| **Pillow (PIL)** | To load and resize images |
| **Matplotlib** | To create an output image with prediction text |
| **HTML (index.html)** | For the user interface (upload and display) |
| **JSON** | To store class labels (like “Tomato___Early_blight”) |

---

## 🚀 End Result:

You now have a working system where:
- **Farmers or anyone** can upload a leaf photo 🧑‍🌾
- The system will **tell them the disease and confidence level** 🔍
- They’ll see a **visual result** with their leaf and prediction ✅

---

Absolutely! Let’s break down your **Plant Disease Prediction** project step by step in very simple and understandable language. We'll go over the two main files: **`model-training.py`** and **`app.py`**.

---

## 🧠 **1. model-training.py** – This script trains the model using plant leaf images

---

### 📦 Step 1: **Extract the Dataset**
```python
dataset_zip_path = r"C:\Users\fg\Desktop\KM\archive.zip"
extract_path = "plantvillage_dataset"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
```

👉 **What’s happening?**
- You are loading a ZIP file that contains plant leaf images.
- If it's not already extracted, it unzips the contents into a folder.

---

### 📁 Step 2: **Set the Base Directory for Images**
```python
base_dir = r"C:\Users\fg\Desktop\KM\plantvillage_dataset\plantvillage dataset\color"
```

👉 **What’s happening?**
- This is the folder that has images of healthy and diseased plant leaves organized in subfolders (each folder is a class, like "Tomato___Late_blight").

---

### 🌀 Step 3: **Prepare the Data with Augmentation**
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    ...
)
```

👉 **What’s happening?**
- This changes the images a bit (rotate, zoom, flip, etc.) to make the model stronger and more flexible.
- It also splits the data: 80% for training, 20% for validation (testing during training).

---

### 📥 Step 4: **Create Data Loaders**
```python
train_generator = train_datagen.flow_from_directory(...)
val_generator = train_datagen.flow_from_directory(...)
```

👉 **What’s happening?**
- These "generators" read images from the folders and prepare them in batches for training the model.

---

### 🧠 Step 5: **Build the Model (CNN)**
```python
model = models.Sequential([
    layers.Conv2D(32, ...),
    layers.MaxPooling2D(...),
    ...
])
```

👉 **What’s happening?**
- You're creating a **Convolutional Neural Network (CNN)** to recognize image patterns.
- It has multiple layers that look at the images, understand features, and finally guess the class (disease type).

---

### 🛠 Step 6: **Compile and Train the Model**
```python
model.compile(...)
history = model.fit(...)
```

👉 **What’s happening?**
- The model is being trained using your images.
- `EarlyStopping`: If the model stops improving, it stops training.
- `ModelCheckpoint`: Saves the best version of the model.

---

### 💾 Step 7: **Save the Model and Class Labels**
```python
model.save('plant_disease_model.h5')
json.dump(class_indices, open('class_indices.json', 'w'))
```

👉 **What’s happening?**
- Saves your trained model so you can use it later.
- Saves the labels (like "Tomato___Late_blight") with numbers so predictions can show readable names.

---

## 🌐 **2. app.py** – This runs the web app using Flask

---

### 🚀 Step 1: **Start the Flask App**
```python
app = Flask(__name__)
```

👉 **What’s happening?**
- This line sets up a web app using Flask, a Python web framework.

---

### 🧠 Step 2: **Load the Model and Class Labels**
```python
model = load_model('plant_disease_model.h5')
with open('class_indices.json') as f:
    class_indices = json.load(f)
```

👉 **What’s happening?**
- It loads the trained model and the labels (class names) into memory so we can make predictions.

---

### 📸 Step 3: **Handle File Upload and Prediction**
```python
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    ...
```

👉 **What’s happening?**
- This function runs when a user visits the site or uploads an image.
- It checks if the file is valid, saves it, and sends it to the prediction function.

---

### 🤖 Step 4: **Predict the Disease**
```python
def predict_disease(image_path):
    img = Image.open(image_path).resize((224, 224))
    ...
    predicted_class = class_names[np.argmax(predictions)]
```

👉 **What’s happening?**
- The uploaded image is resized and converted into a format the model understands.
- The model predicts which disease it thinks the leaf has.
- It also returns a **confidence score**.

---

### 🖼 Step 5: **Show the Result as an Image**
```python
plt.imshow(img)
plt.title(f"{predicted_class} ({confidence})")
```

👉 **What’s happening?**
- This creates a small image with the prediction result written on top.
- Converts the image to base64 so it can be shown on the web page.

---

### 🧪 Final Output
When the user uploads an image:
- The model predicts the disease
- Shows the name + confidence
- Returns an image with the prediction label drawn on it

---

### ✅ Summary

| Part | What it Does |
|------|--------------|
| `model-training.py` | Trains and saves the plant disease detection model |
| `app.py` | Creates a web interface to upload an image and get disease predictions |
| Flask | Makes it possible to run the model in a browser |
| TensorFlow/Keras | Powers the image recognition model |

---

## OUTPUT

![O1](https://github.com/user-attachments/assets/c7525a52-c431-4ccb-8ace-8d98af792aed)

![O2](https://github.com/user-attachments/assets/311389d2-f35c-4f98-9e6f-74a4be0a28ca)

https://github.com/user-attachments/assets/d2b575a5-0304-4a02-830c-7d4bb7b8a68e
