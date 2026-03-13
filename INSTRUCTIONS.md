# Step-by-Step Instructions: Running the Decentralized Cognitive Mesh

Follow these instructions to set up the environment, verify the algorithms, and run the real-time Multi-Node Tracking simulation on your local machine.

---

### Prerequisites
- **Python 3.8+** installed on your system.
- An internet connection for the initial dependency installation (PyTorch, Ultralytics, OpenCV).

---

### Step 1: Open the Project Directory
Open **PowerShell** and navigate to your project folder:
```powershell
cd "c:\Users\Ajay N Mukunth\OneDrive\Desktop\EdgeIntillgence"
```

### Step 2: Create boundary & Activate the Virtual Environment
We need a clean environment to hold our machine learning dependencies.
```powershell
python -m venv venv
.\venv\Scripts\activate
```
*(Your powershell prompt should now have `(venv)` at the beginning of the line).*

### Step 3: Install Dependencies
Run the following command to install the required libraries (PyTorch, YOLOv8 via Ultralytics, OpenCV, and CustomTkinter):
```powershell
pip install -r requirements.txt
```

### Step 4: Verify the Core Math & P2P Logic
Before starting the heavy video processing UI, let's run the lightweight automated tests to prove the P2P networking routing and mathematical matrix fusions work.

Ensure you are still in the active environment, then run:
```powershell
$env:PYTHONPATH="."
python tests\test_p2p.py
```
*You should see outputs verifying the `Store and Forward Test` and the `Fusion Math Test` passing perfectly.*

---

### Step 5: Run the Multi-Node Visual Simulation

We will simulate two independent "cameras" acting as Edge Nodes on your laptop to prove they can track and share intelligence.

**1. Start Node A (Camera 1)**
In your current PowerShell terminal, run:
```powershell
$env:PYTHONPATH="."
python main.py --host 127.0.0.1 --port 5001 --source data\sample_trace.mp4
```
*The "Nerve Center - Node 127.0.0.1:5001" UI window will open.*

**2. Start Node B (Camera 2)**
Open a **brand new PowerShell window**. Navigate to the folder, activate the environment, and start the second node:
```powershell
cd "c:\Users\Ajay N Mukunth\OneDrive\Desktop\EdgeIntillgence"
.\venv\Scripts\activate
$env:PYTHONPATH="."
python main.py --host 127.0.0.1 --port 5002 --source data\sample_trace.mp4
```
*The "Nerve Center - Node 127.0.0.1:5002" UI window will open.*

---

### Step 6: Form the Mesh and Start Tracking

Now that both independent instances are running, we need to tell them about each other to form the "Decentralized Mesh":

1. **On the Node A UI**: Locate the text entry box at the top left. Type `127.0.0.1:5002` and click **Add Peer**.
2. **On the Node B UI**: Locate its text entry box. Type `127.0.0.1:5001` and click **Add Peer**.
3. **Start the Feed**: Click the **Start Stimulus** button on the top right of **both** UIs.

#### What you will see:
- The video feed will begin playing simultaneously in both windows.
- The YOLO model will draw **Red** bounding boxes around newly discovered identities.
- The system will extract their 512-D "Feature Ghost" and securely transmit the mathematical payload over the local mesh.
- You will see logs declaring "`Fused intelligence for [ID]`".
- The bounding box turns **Green** when the identity successfully matched against the collaborative Re-ID memory cache, proving the nodes are tracking *together*.
