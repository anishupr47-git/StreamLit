# NEBULA AI
This web app is designed so that if someone wants to showcase or test an incredibely cool, end to end deep learning training pipeline without actually waiting three days for a server cluster to spin up, or burning through 500 dollars of cloud credits to just to check a UI Layout/
It is an interactive, state of the art streamlit app the bridges the gap between Google Colab and Weights biases gorgeous telemetry tracking.

It's Basically an entire AI training workspace simulation packed into a single, hyper-aesthetic dashboard. It acts as the perfect environment for education, prototyping, demos or to showcase how does ML really work before hand

# Fetures
1. System SMI Console:  It simulates real-time hardware handshakes. Check your PCIe layouts, active memory usage, and get a dynamically updating `nvidia-smi` terminal output for high-end chips like H100s, A100s, or even Apple Silicon.
2. Storage & Staging: Mount virtual drives, upload dataset ZIP files, and watch the YAML class validator parse your directory configuration on the fly.
3. Augmentation Sandbox: Tweak and visualize image enhancements. Adjust sliders for HSV shifts, random noise, and spatial blurs, then see the augmented image side-by-side with original boxes.
4. Backbone Architectures: Design your model architecture. Check real-time parameter sizes, GFLOPs estimates, and generate clean, exportable YOLO network configuration files.
5. AutoML & Sweeps: Spin up hyperparameter sweeps. Run grid searches and render interactive parallel coordinate plots to identify your absolute best parameter configurations.
6. Neural Math Lab: Play out with interactive math. Move IoU sliders (GIoU, DIoU, CIoU) and plot live Focal Loss curves to visualize exactly how class imbalance affects convergence.
7. Roofline Benchmarks: This plots operational intensity vs. memory bounds to see if your simulated hardware is compute-bound or memory-bandwidth-bound.
8. Convergence Training: For this process, click "Start Training" and watch live progress bars, sliding training losses, precision metrics, and dynamically decaying learning rate curves converge in real-time.
9. FastAPI Docker Vault: It generates complete, production-ready python endpoints and optimized Multi-stage Docker files for your newly minted model.
10. Weights Vault Gateway: Serialize your training logs and package your model assets into a clean ZIP archive ready for immediate deployment.
 
 # Tech-Stacl
 1) Streamlit.
 2) NumPy & Pandas
 3) Matplotlib
 4) PyYAML

 # FINALLY
 If someone wants to know how does a core AI system works this is for them. It is an entire AI training workspace which is encapsuled into one single dashboard. Also it also shows how every process is being worked out when someone is developing something from AI. Who enjoys mathematics will also have a good time using this webapp.