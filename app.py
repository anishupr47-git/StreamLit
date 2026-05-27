import os
import sys
import time
import json
import math
import random
import zipfile
import io
import uuid
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import streamlit as st
import numpy as np
import streamlit as st
import pandas as pd

if "stage" not in st.session_state:
    st.session_state["stage"]=0

#Handle pandas, numpy, matplotlibe, yaml imports with graceful fallback alerts
try:
    import pandas as pd
except ImportError:
    st.error("Pandas is not installed. Run `pip install pandas` in your environment")
    pd = None

try:
    import numpy as np
except ImportError:
    st.error("Numpy is not installed. Run `pip install numpy` in your environment")
    np = None

try:
    import numpy as np
except ImportError:
    st.error("Numpy is not installed. Run `pip install numpy` in your environment")
    np = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    st.error("Matplotlib is not installed. Run `pip install matplotlib` in your environment")
    plt = None

try:
    import yaml
except ImportError:
    st.error("PyYAML is not installed. Run `pip install pyyaml` in your environment")
    yaml = None

# SECTION 1: CORE CONSTANT & DEFAULT HARDWARE/DATASET PROFILES

class LabConfigs:
    """Core configuration and hardware options for the AI Workspace Dashboard"""

    SYSTEM_PROVIDERS = [
        "Google Cloud Platform (GCP) - vCompute Instance",
        "Amazon Web Services (AWS) - EC2 GPU Instance",
        "RunPod.io - Secure Cloud Container",
        "Lambda Labs - On-Demand GPU Node",
        "Paperspace Gradient - Dedicated ML Instance",
        "Local Deep Learning Workstation"
    ]

    GPU_PROFILES = {
        "NVIDIA H100 SXM5 80GB" : {
            "vram_gb": 80,
            "cuda_cores": 18432,
            "tensor_cores": 576,
            "architecture": "Hopper",
            "pcie_bandwidth": "Gen5 x16 (128 GB/s)",
            "max_power": "700W",
            "base_temp": "31 C",
            "max_temp": "85",
            "simulated_tflops":67.0,
            "peak_bandwidth_gbs": 3350
        },
        "NVIDIA A100 SXM4 80GB" : {
            "vram_gb": 80,
            "cuda_cores":6912,
            "tensor_cores":432,
            "architecture":"Ampere",
            "pcie_bandwidth":"Gen4 x16 (68 GB/s)",
            "max_power": "400W",
            "base_temp": "34 C",
            "max_temp": "80 C",
            "simulated_tflops": 19.5,
            "peak_bandwidth_gbs": 2039
        },
        "NVIDIA RTX 4090 24GB": {
            "vram_gb": 24,
            "cuda_cores": 16384,
            "tensor_cores": 512,
            "architecture": "Ada Lovelace",
            "pcie_bandwidth": "Gen4 x16 (64 GB/s)",
            "max_power": "450W",
            "base_temp": "38°C",
            "max_temp": "83°C",
            "simulated_tflops": 82.6,
            "peak_bandwidth_gbs": 1008
        },
        "NVIDIA Tesla T4 16GB": {
            "vram_gb": 16,
            "cuda_cores": 2560,
            "tensor_cores": 320,
            "architecture": "Turing",
            "pcie_bandwidth": "Gen3 x16 (16GB/S)",
            "max_power": "70W",
            "base_temp": "35 C",
            "max_temp": "75 C",
            "simulated_tflops": 8.1,
            "peak_bandwidth_gbs": 320
        },
        "Apple M2 Ultra 128GB Unfied" : {
            "vram_gb": 128,
            "cuda_cores": 0,
            "tensor_cores": 32,
            "architecture": "Apple Silicon",
            "pcie_bandwidth": "Unified Memory (800 GB/s)",
            "max_power": "90W",
            "base_temp": "28 C",
            "max_temp": "68 C",
            "simulated_tflops": 27.2,
            "peak_bandwidth_gbs": 800
        }
    }

    YOLO_BACKBONES = {
        "yolov8n.pt":{"params":"3.2M", "flops":"8.7B", "speed_ms":1.2, "description":"Nano model - optimized for edge deploy & mobile devices."},
        "yolov8s.pt":{"params":"11.2M", "flops":"28.6B", "speed_ms":1.8, "description":"Small model - ideal tradeoff between memory usage and speed"},
        "yolov8m.pt":{"params":"25.9M", "flops":"78.9B", "speed_ms":3.6, "description":"Medium model - balanced detector for standard application"},
        "yolov81.pt":{"params":"43.7M", "flops":"165.2B", "speed_ms":5.4, "description":"Large model - high-capacity model designed for complex scenes"},
        "yolov8x.pt":{"params":"68.2M", "flops":"257.8B", "speed_ms":8.2, "description":"Extra Large model - maximum accuracy, high latency applications"}
    }

    DATASET_PRESETS = {
        "Custom Industrial Defects (.yaml)" : {
            "name": "Industrial Anomalies Detection",
            "classes": ["crack", "scratch", "dent", "stain", "misalignment"],
            "img_count": 3420,
            "labels_valid": True,
            "content": (
                "path: ../datasets/industrial_defects\n"
                "train: train/images\n"
                "val: val/images\n"
                "test: test/images\n\n"
                "names:\n"
                " 0: crack\n"
                " 1: scratch\n"
                " 2: dent\n"
                " 3: stain\n"
                " 4: misalignment\n"
            )
        },
        "Autonomous Vehicles Road Objects (.zip)": {
            "name": "AV Road Signs & Vehicles Split",
            "classes": ["car","truck","pedestrian","cyclist","traffic_light","speed_limit"],
            "img_count":12850,
            "labels_valid": True,
            "content": "AV_Road_Staging.zip"
        },
        "Medical Cell Segmentation Config (.yaml)" : {
            "name": "Microscopic Tissue Analysis",
            "classes": ["nuclei","cytoplasm","membrane"],
            "img_count": 1560,
            "labels_valid": True,
            "content": (
                "path: ../datasets/medical_cells\n"
                "train: images/train\n"
                "val: images/val\n\n"
                "names:\n"
                " 0: nuclei\n"
                " 1: cytoplasm\n"
                " 2: membrane \n"
            )
        }
    }
    
# SECTION 2: STYLING USING VANILLA CSS

class StyleSystem:
    """Container moduler vanilla CSS blocks that override default Streamlit Layout blocks"""

    @staticmethod
    def get_custom_css() -> str:
        return """
        <style>
        /*Modern Font Load */
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;700&family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        /*Apply TypoGraphy Elements*/
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #d1d5db;
        }
        h1,h2,h3,h4,h5,h6{
           font-family: 'Outfit', sans-serif;
           font-weight: 700;
           color: #f3f4f6;
           letter-spacing: -0.02em;
        }

        /* Streamlit Override Details*/
        div[data-testid="stAppViewContainer"]{
            background-color: #0c0f16;
            background-image:
                radial-gradient(at 0% 0%, rgba(31,41,55,0.3) 0, transparent 50%),
                radial-gradient(at 100% 0%, rgba(13,148,136,0.08) 0, transparent 40%),
                radial-gradient(at 50% 100%, rgba(99,102,241,0.08) 0, transparent 50%);
            background-attachment: fixed;
        }

        /*Sidebar Styling*/
        div[data-testid="stSidebar"]{
           background-color: #080a0f !important;
           border-right: 1px solid rgba(255,255,255,0.05);
        }

        /*Custom Glowing Badge Systems */
        .badge-container {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }

        .glow-badge {
            font-family: 'Outfit', sans-serif;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 9999px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }

        .badge-active {
            background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.25));
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.4);
            box-shadow: 0 0 15px rgba(16,185,129,0.15);
        }

        .badge-training {
            background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.25));
            color: #10b981;
            border: 1px solid rgba(16,185,129,0.4);
            box-shadow: 0 0 15px rgba(16,185,129,0.15);
        }

        .badge-training {
            background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(217,119,6,0.25));
            color: #f59e0b;
            border: 1px solid rgba(245,158,11,0.4);
            box-shadow: 0 0 15px rgba(245,158,11,0.15);
            animation:pulse-border 2s infinite ease-in-out;
        }

        .badge-idle {
            background: linear-gradient(135deg, rgba(107,114,128,0.15), rgba(75,85,99,0.25));
            color: #9ca3af;
            border: 1px solid rgba(107, 114, 128, 0.3);
        }

        .badge-instance {
            background: linear-gradient(135deg, rgba(6,182,212,0.15), rgba(8,145,178,0.25));
            color: #06b6d4;
            border: 1px solid rgba(6,182,212,0.4);
            box-shadow: 0 0 15px rgba(6,182,212,0.15);
        }

        /* GLASSMORPHIC CONTAINER CARD */

        .glass-card {
            background: rgba(26,31,44,0.55);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.05);
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3);
            transition: border-color 0.3s ease, transform 0.3s ease;
        }

        .glass-card:hover {
            border-color: rgba(6,182,212,0.2);
        }

        .glass-card-header {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 600;
            color: #f3f4f6;
            margin-bottom: 16px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            padding-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Subtitle formatting inside cards */
        .glass-card-subtitle {
            font-size: 0.85rem;
            color: #9ca3af;
            margin-top: -12px;
            margin-bottom: 16px;
        }

        /* Interactive Buttons overrides */
        .stButton>button {
            background: linear-gradient(135deg, #1f2937, #111827) !important;
            color: #e5e7eb !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            padding: 10px 24px !important;
            border-radius: 8px !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
            transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
            box-shadow: 0 4x 6px rgba(0,0,0,0.1) !important;
            width: 100%;
        }

        .stButton>button:hover{
            background: linear-gradient(135deg, rgba(6,182,212,0.1), rgba(99,102,241,0.15)) !important;
            border-color: #06b6d4 !important;
            color: #ffffff !important;
            box-shadow: 0 0 15px rgba(6,182,212,0.35) !important;
            transform: translateY(-2px);
        }

        .stButton>button:active {
            transform: translateY(0px);
        }

        /* Custom Console Output Frame */
        .console-frame {
            background-color: #05070a;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.04);
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.8);
            padding: 16px;
            margin-top: 12px;
            font-family: 'Fira Code' monospace;
            font-size: 0.85rem;
            color: #00ffcc;
            line-height: 1.5;
            overflow-x: auto;
            max-height: 380px;
        }

        /*Metric Blocks Override */
        div[data-testid="stMetricValues"]{
            font-family: 'Outfit',sans-serif;
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
            text-shadow: 0 0 10px rgba(255,255,255,0.1);
        }

        div[data-testid="stMetricLabel"] {
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            color: #9ca3af !important;
        }

        div[data-testid="stMetricDelta"]{
            font-family: 'Fire Code', monospace;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
        }

        /* Input Elements Overrides */
        div[data-baseweb="input"]{
            background-color: #111420 !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,0.8) !important;
            color: #ffffff !important;
            transition: all 0.3s ease !important;
        }

        div[data-baseweb="input"]:focus-within {
            border-color: #06b6df !important;
            box-shadow: 0 0 10px rgba(6,182,212,0.15) !important;
        }

        div[role="listbox"]{
            background-color: #111420 !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
        }

        /* Animations */
        @keyframes pulse-border {
            0% { border-color: rgba(245,158,11,0.4); box-shadow: 0 0 10px rgba(245,158,11,0.15);}
            50% { border-color: rgba(245,158,11,0.8); box-shadow: 0 0 20px rgba(245,158,11,0.35);}
            100% { border-color: rgba(245,158,11,0.4); box-shadow: 0 0 10px rgba(245,158,11,0.15);}
        }

        /*Scrollbar aesthetics */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #080a0f;
        }
        ::-webkit-scrollbar-thumb {
            background: #1f2937;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #374151;
        }

        }

        </style>

            """
# SECTION 3: System Telemetry & CUDA smi Engine

class HardwareTelemetryEngine:
    """Manages Virutal Hardware Specifications, memory consumption logs and SMI generation."""

    @staticmethod
    def generate_nvidia_smi(gpu_type: str, speed: int = 42)-> str:
        """Generates a highly realistic, dyanmaic ASCII nvidia-smi command terminal output."""
        random.seed(seed)

        #Pull specifications based on selete profile
        spec= LabConfigs.GPU_PROFILES.get(gpu_type, LabConfigs.GPU_PROFILES["NVIDIA A100 SXM4 80GB"])
        vram_total = spec["vram_gb"]

        #CALCULATE DYNAMIC LOADS
        
        gpu_util = random.randint(35,95)
        fan_speed= random.randint(45,82)
        power_draw= int(float(spec["max_power"].replace("W","")) * (gpu_util/100.0) + random.randint(15,30))
        power_limit= spec["max_power"]

        temp_val = int(35+(int(spec["max_temp"].replace("C",""))-35)*(gpu_util/100.0) + random.randint(-2,2))

        vram_allocated = int(vram_total * (gpu_util/100.0 * 0.75 + 0.1))

        now = datetime.datetime.now().strftime("%a %b %b %H:%M:%S:%Y")

        #format terminal visual layout
        smi = f"""+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03     CUDA Version: 12.2     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  {gpu_type.ljust(25)} On  | 00000000:00:04.0 Off |                    0 |
| {f"{fan_speed}%".rjust(3)}  {f"{temp_val}C".rjust(4)}    P0         {f"{power_draw}W / {power_limit}".rjust(11)} | {f"{vram_allocated}MiB / {vram_total*1024}MiB".rjust(18)} |   {f"{gpu_util}%".rjust(4)}      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      84102      C   /usr/bin/python3 (JupyterHub)               {int(vram_allocated*0.4)}MiB |
|    0   N/A  N/A      84311      C   ...orch/bin/torch_trainer                   {int(vram_allocated*0.55)}MiB |
|    0   N/A  N/A      91253      C   ...streamlit/bin/app_dashboard              {int(vram_allocated*0.03)}MiB |
+-----------------------------------------------------------------------------------------+"""
        return smi
    
    @staticmethod
    def get_system_topology_summary(gpu_type: str) -> Dict[str, Any]:
        """Calculates system topolgy parameters based on selected acceleration modules"""
        spec= LabConfigs.GPU_PROFILES.get(gpu_type, LabConfigs.GPU_PROFILES["NVIDIA A100 SXM4 80GB"])

        threads_count = 64
        host_ram = 256
        if "RTX 4090" in gpu_type:
            threads_count=32
            host_ram= 64

        elif "Tesla T4" in gpu_type:
            threads_count=32
            host_ram=64

        elif "Apple M2" in gpu_type:
            threads_count=24
            host_ram=128

        return {
            "HOST OS": "Linux 5.15.0-101-generic x86_64",
            "Hardware Accelerator": gpu_type,
            "CUDA Driver / Toolkit": "12.2/12.2.1",
            "Allocated Threads": f"{threads_count} vCPU Cores",
            "System Host Memory": f"{host_ram} GB DDRS System Memory",
            "Simulated Peak Precision Performance": f"{spec['simulated_tflops']} TFLOPS (FP16)"
        }
    
    #SECTION 4 VIRTUAL STORAGE ARCHITECTURE

class VirtualFilesystem:
    """Simulates directory configurations, directory mounting logs, and custom files lists."""
    
    def __init__(self, root_mount: str = "/content/drive/MyDrive/YOLOv8_Workspace"):
        self.root_mount = root_mount.strip()
        self.initialize_state()
        
    def initialize_state(self):
        """Initializes default directory assets structure inside Streamlit state management."""
        if "filesystem" not in st.session_state:
            st.session_state.filesystem = {
                "configs": {
                    "default_dataset.yaml": {"size": "452 B", "type": "yaml", "content": "names: [classA, classB]"},
                    "hyperparameters_preset.yaml": {"size": "1.2 KB", "type": "yaml", "content": "lr0: 0.01\nepochs: 100"}
                },
                "datasets": {
                    "raw_images": {
                        "img_001.jpg": {"size": "240 KB", "type": "image"},
                        "img_002.jpg": {"size": "310 KB", "type": "image"},
                        "img_003.jpg": {"size": "180 KB", "type": "image"}
                    },
                    "annotations": {
                        "img_001.txt": {"size": "120 B", "type": "label"},
                        "img_002.txt": {"size": "98 B", "type": "label"}
                    }
                },
                "runs": {
                    "detect": {
                        "train_exp01": {
                            "weights": {
                                "best.pt": {"size": "84.2 MB", "type": "weights"},
                                "last.pt": {"size": "84.2 MB", "type": "weights"}
                            },
                            "results.csv": {"size": "4.2 KB", "type": "csv"}
                        }
                    }
                }
            }

    def render_tree_html(self, path_dict: Dict[str,Any], depth: int = 0) -> str:
        """Constructs a beautiful visuals nested tree structure in standard markdown format"""
        html_out=""
        indent= "&nbsp;" * (depth*4)
        for key, value in path_dict.items():
            if "size" in value and "type" in value:
                #Leaf elements (File)
                file_icon=""
                if value["type"] == "yaml":
                    file_icon=""
                elif value["type"]=="weights":
                    file_icon=""
                elif value["type"]=="image":
                    file_icon=""
                elif value["type"]=="csv":
                    file_icon= ""
                html_out += f"{indent}{file_icon} <b>{key}</b> <span style='color: #6b7280; font-size: 0.75rem;'>({value['size']})</span><br>"
            else:
                #Node elements (Directory)
                html_out += f"{indent} <span style='color: #38bdf8; font-weight: bold;'>{key}/</span><br>"
                html_out += self.render_tree_html(value,depth + 1)
        return html_out
    

    def add_virtual_file(self, target_folder: str, name: str, size_str: str, file_type: str, content: str = ""):
        """Dynamically registers a mock file in the session """
        parts = target_folder.strip("/").split("/")
        curr = st.session_state.filesystem

        #Navigate nested path mappins safely
        for p in parts:
            if not p:
                continue
            if p not in curr:
                curr[p] = {}

            curr = curr[p]
        curr[name]={"size":size_str, "type":file_type, "content": content}


#SECTION 5: DATASET INSPECTION AND YAML STRUCTURAL VALIDATOR

class DatasetValidator:
    """Validates structural contents of YAML config structures and analyzes dataset distributions."""
    
    @staticmethod
    def parse_yaml_content(yaml_str: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Validates configuration parameters of YOLO configs string syntax."""
        if not yaml:
            return False, None, "PyYAML module is missing. Re-install in host environment."
        
        try:
            parsed = yaml.safe_load(yaml_str)
            if not isinstance(parsed, dict):
                return False, None, "Invalid file format. Contents must load into dictionary properties."
            
            # Look for YOLO structural directories
            required_keys = ["train", "val", "names"]
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                return False, parsed, f"Missing core properties: {', '.join(missing)}"
                
            return True, parsed, "Configuration signature validated successfully."
        except Exception as e:
            return False, None, f"YAML structural exception occurred: {str(e)}"
        
    @staticmethod
    def calculate_synthetic_zip_stats(classes_list: List[str], seed: int = 101) -> Dict[str, Any]:
        """Simulates file-system scanning parameters of uploaded dataset zip partitions."""
        random.seed(seed)
        
        num_classes = len(classes_list)
        total_images = random.randint(1500, 8500)
        train_count = int(total_images * 0.70)
        val_count = int(total_images * 0.20)
        test_count = total_images - (train_count + val_count)
        
        # Generate class frequency spreads using simple power laws
        freq_weights = [1.0 / (i + 1) for i in range(num_classes)]
        sum_weights = sum(freq_weights)
        norm_weights = [w / sum_weights for w in freq_weights]
        
        class_spreads = {}
        for idx, cls in enumerate(classes_list):
            class_spreads[cls] = {
                "Train Annotations": int(train_count * norm_weights[idx] * random.uniform(1.2, 1.8)),
                "Val Annotations": int(val_count * norm_weights[idx] * random.uniform(1.2, 1.8)),
                "Imbalance Ratio": round(norm_weights[idx] / min(norm_weights), 2)
            }
            
        corruptions = random.randint(0, 3)
        unlabeled = random.randint(0, 15)
        
        return {
            "Total Uploaded Archives": 1,
            "Detected Images Split": {"train": train_count, "val": val_count, "test": test_count},
            "Total Labeled Classes": num_classes,
            "Mapped Annotations Logs": class_spreads,
            "Missing Annotation Labels Warnings": unlabeled,
            "Corrupt Image Index Files": corruptions,
            "Quality Status": "🟢 COMPLIANT" if (corruptions == 0 and unlabeled < 5) else "DEGRADED STAGING"
        }
    
#SECTION 6: CONVERGENCE MATHEMATICAL STOCHASTIC SIMULATOR

class DeepLearningPhysicsSimulator:
    """Physics-grade math generators for neural models convergence, loss decays, and scheduling"""

    @staticmethod
    def calculate_lr_decay(
        initial_lr: float,
        epoch: int,
        total_epochs: int,
        schedule_type: str = "Cosine Annealing",
        warmup_epochs: int =5
    ) -> float:
        """Computes current epoch learning rate based on selected schedule type"""
        #Single warmup check
        if epoch <= warmup_epochs:
            return initial_lr * (epoch/float(warmup_epochs))
        
        adjusted_epoch = epoch - warmup_epochs
        adjusted_total = total_epochs - warmup_epochs

        if schedule_type == "Cosine Annealing":
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * adjusted_epoch / adjusted_total))
            return initial_lr * cos_decay
        elif schedule_type == "Step Decay":
            steps = adjusted_epoch // max(1, adjusted_total // 4)
            return initial_lr * (0.1 ** steps)
        elif schedule_type == "Linear Warmup & Plateau":
            return max(initial_lr*0.01, initial_lr * (1.0 - adjusted_epoch / adjusted_total))
        return initial_lr
    
    @staticmethod
    def simulate_epoch_step(
        epoch: int,
        total_epochs: int,
        backbone_scale: float=1.0,
        batch_multiplier: float=1.0,
        current_lr: float = 0.01,
        seed: int = 42        
    ) -> Dict[str, float]:
        """Simulates epoch convergence physics utilizing complex stochastic random walks"""
        #Fix seed dynamically mapped per epoch to gurantee
        np.random.seed(seed+epoch)
        random.seed(seed+epoch)

        progress = epoch / float(total_epochs)

        #Hyperparameters impacts
        scale_dampening = 1.0 + (backbone_scale - 1.0)* 0.15
        batch_dampening = 1.0 - (batch_multiplier - 1.0)*0.05

        #Cosine/Exponential Decay models
        base_decay= math.exp(-3.5*progress)
        noise_variance= 0.06 * math.exp(-2.0*progress)

        #Box loss model
        box_base=1.8*base_decay+0.3
        box_noise = np.random.normal(0, noise_variance *0.8)
        box_loss = max(0.15, (box_base+box_noise)*scale_dampening*batch_dampening)

        #Class Loss Model
        class_base= 2.2 * math.exp(-4.2 * progress) + 0.15
        class_noise = np.random.normal(0, noise_variance *0.8)
        class_loss= max(0.08, (class_base+class_noise)*scale_dampening)

        #Distribution Focal Loss
        dfl_base = 1.2*math.exp(-2.8*progress)+0.4
        dfl_noise = np.random.normal(0, noise_variance * 0.5)
        dfl_loss = max(0.2, (dfl_base + dfl_noise)*batch_dampening)

        total_loss = box_loss + class_loss + dfl_loss

        #Precision & Recall
        map_noise = np.random.normal(0,0.015*math.exp(-1.5*progress))

        #Sigmoid curve matching convergence
        sigmoid_prog = 1.0/(1.0+math.exp(-7.0 * (progress - 0.25)))

        precision = min(0.98, max(0.1, 0.12+0.83 * sigmoid_prog+map_noise))
        recall = min(0.96, max(0.08, 0.09+0.81 * sigmoid_prog + map_noise * 1.2))

        mAP50 = min(0.99, max(0.05, 0.05+0.92 * sigmoid_prog + map_noise * 0.9))
        mAP50_95 = min(0.92, max(0.02, 0.02+0.72 * (sigmoid_prog ** 1.3) + map_noise * 0.7))

        if random.random()<0.03 and epoch > 3 and epoch < total_epochs - 3:
            total_loss *=1.8
            box_loss *=1.6
            mAP50 *= 0.85
            mAP50_95 *=0.8

        return {
            "epoch": float(epoch),
            "box_loss": float(box_loss),
            "class_loss": float(class_loss),
            "dfl_loss": float(total_loss),
            "total_loss": float(total_loss),
            "precision": float(precision),
            "recall": float(recall),
            "mAP50": float(mAP50),
            "mAP50_95": float(mAP50_95),
            "lr": float(current_lr)
        }

#SECTION 7: HYPERPARAMETER TUNING

class HyperparameterSweeper:
    """Manages AutoML dashboard"""

    @staticmethod
    def generate_random_sweep(run_id: int, backbone: str, seed: int = 42) -> Dict[str, Any]:
        """Simulates configuration and final metric score for once run of an AutoML grid search"""
        random.seed(seed+ run_id)

        #Random Pick Hyperparameters
        lr = round(10**random.uniform(-4,-1.5),5)
        batch= random.choice([8,16,32,64])
        optimizer = random.choice(["AdamW","SGD","RMSprop"])
        weight_decay = round(10**random.uniform(-5,-2.5), 6)
        iou_threshold = round(random.uniform(0.45,0.7), 2)

        #Compute scoring metrics
        base_score = 0.55

        #Scale model impact
        backbone_scale = 1.0
        if "yolov8s" in backbone:
            base_score += 0.06
        elif "yolov8m" in backbone:
            base_score +=0.12
        elif "yolov81" in backbone:
            base_score += 0.18
        elif "yolob8x" in backbone:
            base_score += 0.22

        # LR tuning checks
        if 0.0001 <= lr <= 0.015:
            base_score += 0.10
        elif lr > 0.05:
            base_score -= 0.15

        # Optimizer Check
        if optimizer == "AdamW":
            base_score += 0.05

        # Match Weight decay bounds
        if 0.0001 <= weight_decay <= 0.001:
            base_score += 0.04

        base_score += random.uniform(-0.04, 0.04)

        final_mAP50 = min(0.99, max(0.12, base_score))
        final_mAP50_95 = min(0.92, max(0.06, final_mAP50 * 0.72))
        final_loss = max(0.3,2.5 - (final_mAP50 * 2.1))

        return{
            "Sweep Run": f"run_{run_id:03d}",
            "Backbone": backbone,
            "LR": lr,
            "Batch Size": batch,
            "Optimizer": optimizer,
            "Weight Decay": weight_decay,
            "IoU Threshold": iou_threshold,
            "mAP50": round(final_mAP50, 4),
            "mAP50-95": round(final_loss, 4),
            "Status": "COMPLETED" if final_loss < 2.0 else "DIVERGED"
        }
    

#WEIGHTS VAULT & MODEL EXPORTER INTERFACE

class WeightsVaultExporter:
    """Manages weights storage mapping, CSV report summaries, and memory compilation zip generation."""
    
    @staticmethod
    def generate_results_csv(history: List[Dict[str, float]]) -> str:
        """Converts local python dictionaries logs to a downloadable comma-separated string."""
        if not history:
            return ""
        
        headers = ["epoch", "box_loss", "class_loss", "dfl_loss", "total_loss", "precision", "recall", "mAP50", "mAP50_95", "lr"]
        csv_out = ",".join(headers) + "\n"
        
        for ep in history:
            row = [str(ep.get(h, 0.0)) for h in headers]
            csv_out += ",".join(row) + "\n"
            
        return csv_out

    @staticmethod
    def package_vault_zip(
        yaml_config: str,
        results_csv: str,
        best_accuracy: float,
        backbone_name: str
    ) -> bytes:
        """Builds in-memory zip bytes container holding all final training logs and weights."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            # 1. Add model card
            model_card = f"""# MODEL TRAINING CARD - {datetime.datetime.now().strftime("%Y-%m-%d")}
 Deep Learning Workspace Staging Sandbox.

## Performance Profile
- Model Backbone: {backbone_name}
- Final Peak Accuracy (mAP50-95): {best_accuracy * 100:.2f}%
- Serialization Time: {datetime.datetime.now().strftime("%H:%M:%S")}
- System Target Platform: CUDA Accelerated Linux Virtual Node

## Compiled Weights Contents
- `best.pt` - Peak accuracy validation checkpoint.
- `last.pt` - Final training epoch state checkpoint.
- `dataset.yaml` - Class index dataset mappings config.
- `training_progress_report.csv` - Epoch-by-epoch loss statistics.
"""
            zip_file.writestr("model_card.md", model_card)
            zip_file.writestr("dataset.yaml", yaml_config)
            zip_file.writestr("training_progress_report.csv", results_csv)
            zip_file.writestr("weights/best.pt", f"SERIALIZED_TENSOR_WEIGHTS_DUMMY_BINARY_DATA_{uuid.uuid4().hex}".encode('utf-8'))
            zip_file.writestr("weights/last.pt", f"SERIALIZED_TENSOR_WEIGHTS_DUMMY_BINARY_DATA_{uuid.uuid4().hex}".encode('utf-8'))
            
        return zip_buffer.getvalue()
    
    @staticmethod
    def get_compilation_details(target_format: str, model_name: str) -> Dict[str, Any]:
        """Provides simulated details metrics of compiled models exports profiles"""
        if target_format == "ONNX (FP16)":
            return {
                "Command Run": f"yolo export model={model_name} format=onnx half=True opset=12",
                "Export File Name": f"{model_name.replace('.pt','')}.onnx",
                "Size Reduction": "48.2% Reduction (FP32 -> FP16)",
                "Export Latency": "14.2 (Completed)",
                "Edge Target Precision": "FP16 Floating Point Half Precision",
                "Inference Latency (RTX 4090)": "0.45 ms / image",
                "Optimization Pipeline": "Pruning: Disabled | Layer Fusion: Enabled"
            }
        elif target_format == "TensorRT (INT8 Quantized)":
            return {
                "Command Run": f"yolo export model={model_name} format=engine int8=True device=0 workspace=4",
                "Export File Name": f"{model_name.replace('.pt','')}.mlmodelc",
                "Size Reduction":"49.5% Reduction",
                "Export Latency":"34.1s (Completed)",
                "Edge Target Precision": "FP16 Apple CoreML Engine M-Series Target",
                "Inference Latency (M2 Ultra)": "1.12 ms / image",
                "Optimization Pipeline": "Metal Performance Shaders Layer Pruning"

            }
        return {}
    
#ADVANCED NEURAL MATH PLAYGROUND & PURE PYTHON ALGORITHMS

class NeuralMathPlayground:
    """Implements educational, fully operational pure-Python math algorithms for ML tasks"""

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculates dynamic Intersection-over-Union (IoU) ratio between two bouding boxes"""
        x_left =max(box1[0], box2[0])
        y_top=max(box1[1],box2[1])
        x_right= min(box1[2],box2[2])
        y_bottom= min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2]-box2[0])*(box2[3]-box2[1])

        union_area = float(area_box1 + area_box2 - intersection_area)
        if union_area <= 0.0:
            return 0.0
        
        return intersection_area/union_area
    
    @staticmethod
    def calculate_giou(box1:List[float],box2: List[float]) -> float:
        """Calculates Generalized Interesection-over-union (GIoU)"""
        iou = NeuralMathPlayground.calculate_iou(box1, box2)

        c_x1 = min(box1[0], box2[0])
        c_y1 = min(box1[1], box2[1])
        c_x2 = max(box1[2],box2[2])
        c_y2 = max(box1[3], box2[3])

        c_area = (c_x2 - c_x1) * (c_y2-c_y1)

        area_box1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area_box2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        interesction = iou * (area_box1 + area_box2) / (1.0+iou) if iou > 0 else 0
        union = area_box1 + area_box2 - interesction

        if c_area <=0:
            return iou
        
        return iou - (c_area - union) / c_area
    

    @staticmethod
    def calculate_diou(box1: List[float], box2: List[float]) -> float:
        """Calculates Distance Intersection Over Union(DIoU)"""
        iou = NeuralMathPlayground.calculate_iou(box1,box2)
        if iou <=0:
            return 0.0
        
        c1_x = (box1[0]+box1[2])/2.0
        c1_y = (box1[1]+box1[3])/2.0
        c2_x = (box2[0]+box2[2])/2.0
        c2_y =(box2[1]+box2[3])/2.0

        d_center = (c1_x - c2_x)**2+(c1_y-c2_y)**2

        c_x1= min(box1[0], box2[0])
        c_y1= min(box1[1], box2[1])
        c_x2= max(box1[2],box2[2])
        c_y2= max(box1[3], box2[3])
        diagonal = (c_x2-c_x1)**2 + (c_y2-c_y1)**2

        if diagonal <=0:
            return iou
        
        return iou - d_center / diagonal
    

    @staticmethod
    def execute_nms(
        boxes: List[List[float]],
        scores: List[float],
        iou_threshold: float = 0.5
    ) -> List[int]:
        """Calculates valid indices by performing Non-Maximum Suppression on bounding detections"""
        if not boxes:
            return[]
        
        indicies = list(range(len(scores)))
        indicies.sort(key=lambda i: scores[i], reverse=True)

        keep_indices = []
        while indicies:
            curr_idx = indicies.pop(0)
            keep_indices.append(curr_idx)

            remaining_indices = []
            for idx in indicies:
                iou = NeuralMathPlayground.calculate_iou(boxes[curr_idx], boxes[idx])
                if iou < iou_threshold:
                    remaining_indices.append(idx)

            indicies = remaining_indices

        return keep_indices
    
    @staticmethod
    def compute_confusion_matrix(classes: List[str], seed: int =42) -> pd.DataFrame:
        """Simlulates statistival parameters of a confusion matrix"""
        random.seed(seed)
        num_classes= len(classes)
        matrix = np.zeros((num_classes, num_classes), dtype=int)

        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    matrix[i,j] = random.randint(85,250)
                else :
                    matrix[i,j] = random.randint(0,15) if random.random() < 0.4 else 0

        return pd.DataFrame(matrix, index=classes, columns=classes)
    
    @staticmethod
    def calculate_precision_recall_points(seed: int = 42) -> Tuple[List[float], List[float]]:
        """Generates coordinate arrays for plotting a standard precision"""
        np.random.seed(seed)
        recalls = np.linspace(0.0, 1.0, 100)
        precision = 1.0 - (recalls ** 3) * 0.45 + np.random.normal(0,0.015,len(recalls))
        precision= np.clip(precision,0.0,1.0)
        precision= np.maximum.accumulate(precision[::-1])[::-1]
        return list(recalls), list(precision)
        
#SECTION 10: OPTIMIZER SPECIFIC
class OptimizerTheoryMatrix:
   """Houses mathematical parameters, description, and learning dynamics for deep learning optimizers"""

DETAILS = {
    "AdamW": {
        "title": "Adam with Decoupled Weight Decay (AdamW)",
        "math": (
            "g_t= \\nabla f(w_t) \\\\\n"
            "m_t= \\beta_1 m_{t-1}+(1-\\beta_1) g_t \\\\\n"
            "v_t= \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2 \\\\\n"
            "\\hat{m}_t= m_t / (1-\\beta_1^t) \\\\\n"
            "\\hat{v}_t= v_t / (1-\\beta_2^t) \\\\\n"
            "w_{t+1}= w_t - \\eta \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}+\\epsilon}+\\lambda w_t \\right)"

        ),
        "description": (
            "AdamW solves the weight decay coupling issue in standard Adam. It directly  applies"
            "weight decay penalty to the parameters rather than incorporating it into momentum calculations"
            "This achieves superior generalization on deep transformer layers and object detection backbones"
        ),
        "parameters" : {
            "Beta 1 (Momentum Decay)": "0.9 (Standard)",
            "Beta 2 (Variance Decay)": "0.999 (Standard)",
            "Epsilon (Numerical Bounds)": "1e-8 (Stabilizer)",
            "Weight Decay (L2 Decoupled)": "0.01 (Configurable)"
        }
    },
    "SGD": {
        "title": "Stochastic Gradient Descent with Momentum (SGD-M)",
        "math": (
            "g_t=\\nabla f(w_t) \\\\\n"
            "v_t= \\gamma v_{t-1} + \\eta g_t \\\\\n"
            "w_{t+1} = w_t - v_t"
        ),
        "description": (
            "Stochastic Gradient Descent with classical momentum accelerates gradient descent vectors"
            "along directions of persistent convergence. This dampens oscillating noise vectors"
            "Highly effective for classical CNN layouts and ResNet training models where batch sizes are large"
        ),
        "parameters":{
            "Momentum Damping (Gamma)": "0.93 (Classical)",
            "Nesterov Acceleration": "Enabled (Accelerates slopes)",
            "Weight Decay (L2 Regularization)": "5e-4"
        }
    },
    "RMSprop": {
        "title": "Root Mean Square Propagation (RMSprop)",
        "math": (
            "g_t= \\nabla f(w_t) \\\\\n"
            "v_t= \\gamma v_{t-1} + \\eta g_t \\\\\n"
            "w_{t+1} = w_t - v_t"
        ),
        "description": (
            "Stochastic Gradient Descent with classical momentum accelerates gradient descent vectors"
            "along directions of persistent convergence. This dampens oscillating noise vectors"
            "Highly effective for classical CNN layouts and ResNet training models where batch sizes are large."
        ),
        "parameters": {
            "Momentum Dampening (Gamma)": "0.93 (Classical)",
            "Nesterov Acceleration": "Enabled (Accelerates slopes)",
            "Weight Decay (L2 Regularization)": "5e-1"
        }
    },
    "RMSprop": {
        "title": "Root Mean Square Propagation (RMSprop)",
        "math": (
            "g_t= \\nabla f(w_t) \\\\\n"
            "v_t= \\alpha v_{t-1} + (1-\\alpha) g_t^2 \\\\\n"
            "w+{t+1}= w_t - \\frac{\\eta}{\\sqrt{v_t}+ \\epsilon} g_t"
        ),
        "description": (
            "RMSprop restricts gradient osciallations in vertical directions by dividing the gradient"
            "by a running average of its recent magnitudes. Introduced by Geoff Hinton, it works well"
            "for recurrent networks and complex non-stationary objective funtions"
        ),
        "parameters": {
            "Smoothing Alpha": "0.99 (Standard)",
            "Epsilon (Numerical Bounds)": "1e-8",
            "Weight Decay": "0.0 (Optional)"
        }
    }  
}
OptimizerTheoryMatrix.DETAILS = DETAILS

#Image Authenticator
class ImageAugmentationSimulator:
    """Applies simulated pixel operations and geometric transforms to mock images."""
    
    @staticmethod
    def generate_mock_image(size: int = 120, seed: int = 42) -> np.ndarray:
        """Generates a synthetic spatial image representation containing mock bounding shapes."""
        np.random.seed(seed)
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :] = [18, 24, 38]
        img[30:70, 40:80] = [30, 58, 138]
        img[90:100, 15:25] = [200, 80, 80]
        return img

    @staticmethod
    def apply_noise(img: np.ndarray, variance: float) -> np.ndarray:
        """Applies stochastic gaussian noise artifacts directly to pixel arrays."""
        if variance <= 0:
            return img.copy()
            
        noise = np.random.normal(0, variance * 255, img.shape)
        noisy = img.astype(float) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_blur(img:np.ndarray, kernel_size: int) -> np.ndarray:
        """Applies horizontal spatial blur by average shifting pixels"""
        if kernel_size <=1:
            return img.copy()
        
        blurred = img.copy().astype(float)
        for c in range(3):
            for i in range(1, img.shape[0] -1):
                for j in range(1, img.shape[1] -1):
                    blurred[i,j,c]=np.mean(img[i-1:i+2, j-1:j+2, c])

        return np.clip(blurred, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_contrast(img: np.ndarray, factor:float) -> np.ndarray:
        """Scales intensity spreads relative to median baseline channels"""
        mean=128.0
        adjusted= mean+factor*(img.astype(float)-mean)
        return np.clip(adjusted,0,255).astype(np.uint8)
    

#Deployment script generator

class DeploymentScriptGenerator:
    """Compiles clean operational code templates for deployment servers,fastapis, and dockers"""

    @staticmethod
    def generate_fast_api_code(model_name:str, classes: List[str]) -> str:
        """Returns standard python FastAPI code for web hosting inferences requests"""
        classes_str= ",".join([f"{c}"for c in classes])
        return f"""# ==============================================================================
# FastAPI Model Prediction Server API
# Deep Learning Workspace deployment compiler
# ==============================================================================
import time
import io
from typing import List,Dict,Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
import onxruntime as ort

app = FastAPI(
    title="YOLOv8 Edge Prediction Service",
    description="FastAPI server serving custom trained weights target architecture",
    version="1.0.0"
)

MODEL_PATH = "runs/detect/train_exp01/{model_name.replace('.pt','')}.onnx"open
classes_map = [{classes_str}]

try:
   session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
   input_name = session.get_inputs()[0].NameError
   print(f"[SUCCESS] Inference Session locked on target devices: {{ort.get_device()}}")
except Exception as e:
   session= None
   print(f"[WARNING] Failed to load model weights: {{str(e)}}. Running in fallback simulation mode")

class BBoxDetection(BaseModel):
   class_name: str
   confidence: float
   bbox: List[float]

class ServerResponse(BaseModel):
   inference_ms: float
   device_status: str
   detections: List[BBoxDetection]

@app.get("/")
def health_check() -> Dict[str, str]:
    return {{"status": "healthy", "engine": "ONNXRuntime", "model": "{model_name}}}

@app.post("/predict", response_model=ServerResponse)
async def predict_image(file: UploadFile = File(...)) -> ServerResponse:
   start_time = time.time()

   if not file.content_type.startswith("image/"):
      raise HTTPSException(status_code=400, detail="Invalid target file format. Must upload standard image")

    try:
       img_bytes = await file.read()
       image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

       img_resized = image.resize((640,640))
       img_array = np.array(img_resized).astype(np.float32) / 255.0
       img_transposed = np.transpose(img_array, (2,0,1))[np.newaxis, :, :, :]

       detections = []
       if session is not None:
          outputs = session.run(None, {{input_name: img_transposed}})
          detections.append(BBoxDetection(
             class_name=classes_map[0] if classes_map else "target_anomaly",
             confidence=0.88,
             bbox=[120.5,98.2,340.1,412.5]
            ))
        else:
            time.sleep(0.015)
            detections.append(BBoxDetection(
                class_name=classes_map[0] if classes_map else "mock_object",
                confidence=0.91,
                bbox=[45.0,50.0,180.0,220.0]
            ))

        latency = (time.time() - start_time) * 1000.0
        return ServerResponse(
            inference_ms=round(latency,2),
            device_status="GPU_ACCELERATED" if session else "CPU_EMULATOR",
            detections=detections
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference exception occured: {{str(exc)}}")

if __name__ == "__main":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0", port=8000)
"""
    @staticmethod
    def generate_dockerfile(model_name:str) -> str:
        """Returns target production Dockerfile configurations mapping CUDA layers."""
        return f"""# ==============================================================================
# Production Dockerfile for deep learning edge APIs
# ==============================================================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=nointeractive \\
    PYTHONNUMBERBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3-pip \\
    python3-dev \\
    ffmpeg \\
    libsm \\
    libxext6 \\
    git \\
    curl \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir \\
    fastapi \\
    uvicorn \\
    pydantic \\
    numpy \\
    opencv-python-headless \\
    pillow \\
    onnxruntime-gpu

COPY runs/detect/train_exp01/{model_name.replace('.pt','')}.onnx ./runs/detect/train_exp01/
COPY server_api.py .

EXPOSE 8000

CMD ["python3", "server_api.py"]
"""
    
    @staticmethod
    def generate_pytorch_inference(model_name: str, classes: List[str]) -> str:
        classes_str = ", ".join([f" '{c}'" for c in classes])
        return f"""# ==============================================================================
# PyTorch Native Inference Execution Script
# Deep Learning Workspace
# ==============================================================================
import torch
import cv2
import numpy as np
from ultralytics import YOLO

#1. Initialize custom backbone target weights
model = YOLO("{model_name}")
classes_list = [{classes_str}]

def run_image_prediction(image_path: str):
    # Read spatial matrix channels
    image = cv2.imread(image_path)
    if image is None:
       raise ValueError(f"Target image path is unreadable: {{image_path}}")

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    results = model(image, device="cuda" if torch.cuda.is_available() else "cpu")
    end_time.record()
    
    torch.cuda.synchronize()
    latency = start_time.elapsed_time(end_time)
    print(f"[INFO] Prediction completed in {{latency:.2f}} ms.")

    #2. Iterate detections bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Anchor coordinates formats: [xmin,ymin,xmax,ymax]
            coords = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_idx = int(box.cls[0].item())

            print(f"Detected Class: {{classes_list[cls_idx] if cls_idx < len(classes_list) else 'unknown'}} | Conf: {{conf*100:.1f}}% | Box: {{coords}}")

            #Write annotated bounding boxes back to spatial frame
            cv2.rectangle(
                image,
                (int(coords[0]), int(coords[1])),
                (int(coords[2]), int(coords[3])),
                (0,255,0),2
            )

        cv2.imwrite("annoated_prediction.jpg", image)
        print("[SUCCESS] Annoated visual saved to path: 'annoated_prediction.jpg'")

if __name__ == "__main__":
    run_image_prediction("test_sample.jpg")
"""
    @staticmethod
    def generate_cpp_inference(model_name:str, classes: List[str])->str:
        classes_array = ", ".join([f'"{c}"' for c in classes])
        return f"""// ==============================================================================
// C++ OpenCV DNN Prediction Engine
// Deep Learning Workspace
// ==============================================================================
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

const std::vector<std::string> classes_map = {{{classes_array}}};

int main(int argc, char** argv) {{
    std::string model_path = "runs/detect/train_exp01/{model_name.replace('.pt', '')}.onnx";
    std::string img_path = "test_sample.jpg";

    //Load optimized ONNX graph network
    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);

    // Initialize CUDA graphics execution modules
    if (cv::cuda::getCudaEnabledDeviceCount()>0){{
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dn::DNN_TARGET_CUDA);
    std::cout << "[SUCCESS] C++ DNN backend established on active GPU registers." <<std::endl;
    }} else {{
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "[INFO] Fallback established on CPU core execution" <<std::endl;
    }}

    cv::Mat frame = cv::imread(img_path);
    if (frame.empty()) {{
        std::cerr << "Error: Target image file empty or unreadable" << std::endl;
        return -1;
    }}

    // Pre-processing pipeline: Blob scaling parameters (640x640)
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);
    
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    
    std::cout << "[SUCCESS] Graph inference executed. Parsing bounding boxes maps." << std::endl;
    // Bbox parsing loops...
    return 0;
}}
"""


#COMPREHENSIVE ADVANCED LOGGIN MODULE

class AdvancedLoggerManager:
    """Manages workspace loggin records, categorizing logs and supporting filters"""

    @staticmethod
    def generate_system_logs(seed: int=42) -> List[str]:
        """Assembles list of highly realistic system logs."""

        random.seed(seed)
        logs=[]
        now= datetime.datetime.now()

        log_templates = [
            ("[INFO]", "Loading ML environment variables context..."),
            ("[INFO]", "Python runtime initialzied: version 3.10.12 Ubuntu"),
            ("[INFO]", "Streamlit server core active running on thread 0x7fa281"),
            ("[INFO]", "GPU drivers queried. Found active CUDA device index 0."),
            ("[SUCCESS]", "CUDA connection handshake verified. NVIDIA-SMI status active"),
            ("[INFO]", "Virtual filesystem mount requsted..."),
            ("[SUCCESS]", "Drive mounted successfully at root path: /content/drive/MyDrive"),
            ("[INFO]", "dataset.yaml configuration parsed. Mapped classes parameters verified"),
            ("[INFO]", "Hyperparameter optimization bounds set successfully"),
            ("[INFO]", "Starting deep learning model execution run..."),
            ("[INFO]", "AdamW weight decay parameters cached to GPU kernel"),
            ("[SUCCESS]", "Training loop completed. Saved checkpoint files best.pt."),
            ("[INFO]", "ONNX model compilation request detected"),
            ("[SUCCESS]", "ONNX model saved successfully. Inference engine cached")
        ]

        for idx, (level, msg) in enumerate(log_templates):
            delta = datetime.timedelta(minutes=idx*2, seconds=random.randint(0,59))
            t_str = (now-delta).strftime("%Y-%m-%d %H:%M:%S")
            logs.append(f"{t_str} {level.ljust(9)} {msg}")

        return logs
    
#ADVANCED DEEP LEARNING THEORY MATRIX

class AdvancedDeepLearningTheory:
    """Educational class compiling key deep learning concepts, loss equations, and matrix"""

    @staticmethod
    def get_focal_loss_explanation() -> Dict[str, Any]:
        return {
            "title": "Focal Loss for Dense Object Detection",
            "equation": "FL(p_t) = -\\alpha_t (1-p_t)^\\gamma\\log(p_t)",
            "description": (
                "Focal Loss dynamically scales standard Cross-Entropy based on prediction confidence"
                "The modulating factor (1-p_t)^gamma suppresses gradient contribution from easy negative background"
                "samples during classification runs, concentrating training purely on hard foreground anomalies"
            ),
            "hyperparameters":{
                "Gamma (Focusing parameters)": "2.0 (Balances easy vs hard samples)",
                "Alpha (Balance parameters)": "0.25 (Balances foreground vs background class counts)"
            }
        }
    
    @staticmethod
    def get_dfl_explanation() -> Dict[str, Any]:
        return {
            "title": "Distribution Focal Loss (DFL)",
            "equation": "DFL(S_i, S_{i+1})= -((y_{i+1}-y) \\logs(S_i)+(u - y_i) \\logs(S_{i+1}))",
            "description": (
                "Standard regression optimizes targets as absolute coordinates Dirac delta distributions"
                "DFL maps continious boundaries coordinate regions as complete probability distributions"
                "This significantly improves accuracy boundaries in ambigious situations, e.g., occulsion or blur"
            )
        }
    
#SECTION 15: COMPUTE VS MEMORY ROOFLINE

class RooflineBenchmarkEngine:
    """Calculates operational intensities and compiles Roofline graphs for chosen hardware configurations"""

    @staticmethod
    def calculate_operational_intensity(flops: float, memory_bytes: float) -> float:
        """Returns FLOPs per Byte computational density"""

        if memory_bytes <=0:
            return 0.0
        return flops / memory_bytes
    
    @staticmethod
    def get_roofline_profile(gpu_type: str, flops_g: float = 28.6) -> Dict[str, Any]:
        """Calculates system bottlenecks based on hardware characteristics"""
        spec = LabConfigs.GPU_PROFILES.get(gpu_type, LabConfigs.GPU_PROFILES["NVIDIA A100 SXM4 80GB"])

        peak_tflops = spec["simulated_tflops"]
        peak_bandwidth = spec.get("peak_bandwidth_gbs", 2039.0)

        knee_point = (peak_tflops * 1000.0) / peak_bandwidth

        flops_total= flops_g * 1e9
        model_size_bytes= 22.4 * 1e6

        op_intensity = flops_total / model_size_bytes

        is_compute_bound = op_intensity > knee_point
        bottleneck = "COMPUTE-BOUND" if is_compute_bound else "MEMORY-BOUND (BANDWIDTH RESTRICTED)"

        max_perf = min(peak_tflops, (op_intensity * peak_bandwidth) / 1000.0)

        return {
            "Peak FLOPS Target": f"{peak_tflops} TFLOPS",
            "Peak Memory Bandwidth": f"{peak_bandwidth} GB/S",
            "Hardware Knee Point": f"{knee_point:.2f} FLOPs/Byte",
            "Calculated Model Intensity": f"{op_intensity:.2f} FLOPs/Byte",
            "Caclulated Model Intensity": f"{op_intensity:.2f} FLOPs/Byte",
            "System Bottleneck Status": bottleneck,
            "Max Achievable Performance":f"{max_perf:.2f} TFLOPS"
        }
    
#PURE PYTHON IN BROWSER NEURAL WEIGHTS OPTIMIZER

class NeuralLayerOptimizerSimulator:
    """Provides pure-Python simulator of weights tensor distributions and custom optimizer steps"""

    @staticmethod
    def initialize_layer_weights(dim: int=16, init_type: str= "Xavier Normal", seed: int=42) -> np.ndarray:
        """Intializes a weight matrix matching selected neural parameters"""
        np.random.seed(seed)
        if init_type == "Xavier Normal":
            std = math.sqrt(2.0/(dim+dim))
            return np.random.normal(0, std, (dim, dim))
        elif "He" in init_type:
            std= math.sqrt(2.0/dim)
            return np.random.normal(0, std, (dim, dim))
        else:
            return np.random.normal(0,1.0,(dim,dim))
        
    @staticmethod
    def step_matrix_optimization(
        weights: np.ndarray,
        gradients: np.ndarray,
        velocity: np.ndarray,
        learning_rates: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates SGD-with-Momentum step update over active weight tensors"""
        weight_decayed = weights * (1.0 - learning_rates * weight_decay)
        new_velocity = momentum * velocity + gradients
        new_weights = weight_decayed - learning_rates * new_velocity

        return new_weights, new_velocity
    
#YOLOV8 ADVANCED FULL PYTORCH CODE GENERATOR

class YoloPyTorchCodeGenerator:
    """Assembles operational PyTorch layer codes for custom detection model components"""

    @staticmethod
    def generate_conv_module() -> str:
        return """import torch
import torch.nn as nn

class Conv(nn.Module):
    \"\"\"Standard Convolution layer containing Conv2d + BatchNorm2d + SiLU activation module.\"\"\"
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        padding = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, padding, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
"""


    @staticmethod
    def generate_c2f_module() -> str:
        return """class C2f(nn.Module):
    \"\"\"YOLOv8 custom C2f (CSP Bottleneck with 2 convolutions) layer definition.\"\"\"
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2*e)
        self.cv1 = Conv(c1, 2*self.c, 1, 1)
        self.cv2 = Conv((2+n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
    
    def forward(self,x):
        y= list(self.cv1(x).chunk(2,1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y,1))
"""

#TRANSFORMS MATRIX ATTENTION SIMULATOR

class SelfAttentionSimulator:
    """Generates synthetic Query-Key-Value projection matrices and plots Self-Attention Maps"""

    @staticmethod
    def compute_attention_map(seq_len: int=16, seed: int=42) -> np.ndarray:
        """Simulates self-attention dot product projection score layers"""
        np.random.seed(seed)

        #1. Generate dummy Query and key tensors matrices
        q = np.random.normal(0,0.4,(seq_len, 8))
        k = np.random.normal(0, 0.4, (seq_len, 8))

        #2. Compute Q K^T scaled dot product attention parameters
        scaled_dot = np.matmul(q, k.T) / math.sqrt(8.0)

        #3. Apply softmax scaling across horizontal axes
        exp_dot = np.exp(scaled_dot - np.max(scaled_dot, axis=-1, keepdims=True))
        softmax_attention = exp_dot / np.sum(exp_dot, axis=-1, keepdims=True)

        return softmax_attention
    
#STREAM MEMORY BANDWIDTH SIMULATOR

class StreamMemorySimulator:
    """Generates synthetic STREAM benchmarks memory metrics (Copy, Scale, Add, Triad)"""

    @staticmethod
    def get_stream_bandwidths(gpu_type: str) -> Dict[str, float]:
        """Provides simulated STREAM benchmark execution bandwidths based on GPU architectures"""
        spec = LabConfigs.GPU_PROFILES.get(gpu_type, LabConfigs.GPU_PROFILES["NVIDIA A100 SXM4 80GB"])
        peak_bw = spec.get("peak_bandwidth_gbs", 2039.0)

        #Replicate classical STREAM scaling dampening factor
        dampening = 0.88

        return {
            "STREAM Copy (GB/s)": round(peak_bw * dampening * 0.94, 2),
            "STREAM Scale (GB/s)": round(peak_bw * dampening * 0.92, 2),
            "STREAM Add (GB/s)": round(peak_bw * dampening * 0.96, 2),
            "STREAM Triad (GB/s)": round(peak_bw * dampening, 2)
        }
    
#MAIN APPLICATION SETUP AND MULTI STAGE ROUTING

def main():
    """Initializes workspace stages, handles widget layout rendering and triggers simulation sweeps"""
    global fs

    #1. Setup Streamlit page definitions
    st.set_page_config(
        page_title="Deep Learing Lab Workspace",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    #2. Inject CSS Styling Block
    st.markdown(StyleSystem.get_custom_css(), unsafe_allow_html=True)

    #3. Session State Key-Value Initializations
    if "stage" not in st.session_state:
        st.session_state.stage = 1
    if "hardware_locked" not in st.session_state:
        st.session_state.hardware_locked = False
    if "selected_gpu" not in st.session_state:
        st.session_state.selected_gpu = "NVIDIA A100 SXM4 80GB"
    if "system_provider" not in st.session_state:
        st.session_state.system_provider = "Google Cloud Platform (GCP) - vCompute Instance"
    if "nvidia_smi_log" not in st.session_state:
        st.session_state.nvidia_smi_log = ""
    if "drive_mounted" not in st.session_state:
        st.session_state.drive_mounted = False
    if "drive_path" not in st.session_state:
        st.session_state.drive_path = "/content/drive/MyDrive/YOLOv8_Training_Lab"
    if "uploaded_dataset_type" not in st.session_state:
        st.session_state.uploaded_dataset_type = None
    if "dataset_filename" not in st.session_state:
        st.session_state.dataset_filename = ""
    if "dataset_yaml_data" not in st.session_state:
        st.session_state.dataset_yaml_data = ""
    if "classes_detected" not in st.session_state:
        st.session_state.classes_detected=[]
    if "active_dataset_preset" not in st.session_state:
        st.session_state.active_dataset_preset = None

    #Visual Image Augmentation parameters
    if "aug_noise" not in st.session_state:
        st.session_state.aug_noise=0.0
    if "aug_blur" not in st.session_state:
        st.session_state.aug_blur=1
    if "aug_contrast" not in st.session_state:
        st.session_state.aug_contrast = 1.0

    #Backbone Custom Designer
    if "custom_backbone_layers" not in st.session_state:
        st.session_state.custom_backbone_layers = [
            {"layer_id":1, "type":"Conv2d", "out_channels":64, "kernel_size":3, "stride":2},
            {"layer_id":2, "type":"BatchNorm2d", "out_channels": 64, "kernel_size": "-", "stride": "-"},
            {"layer_id":3, "type": "SiLU", "out_channels": "-", "kernel_size": "-", "stride": "-"}
        ]

    #BROWSER pure-python matrix visualizer
    if "visual_weights_matrix" not in st.session_state:
        st.session_state.visual_weights_matrix = NeuralLayerOptimizerSimulator.initialize_layer_weights()
    if "visual_weights_velocity" not in st.session_state:
        st.session_state.visual_weights_velocity = np.zeros((16,16))
    if "layer_init_method" not in st.session_state:
        st.session_state.layer_init_method= "Xavier Normal"
    

    #Hyperparameters selection
    if "model_backbone" not in st.session_state:
        st.session_state.model_backbone = "yolov8s.pt"
    if "batch_size" not in st.session_state:
        st.session_state.batch_size = 16
    if "learning_rate" not in st.session_state:
        st.session_state.learning_rate= 0.01
    if "epochs_count" not in st.session_state:
        st.session_state.epochs_count =20
    if "optimizer_type" not in st.session_state:
        st.session_state.optimizer_type = "AdamW"
    if "lr_decay_type" not in st.session_state:
        st.session_state.lr_decay_type= "Cosine Annealing"
    
    #Active training logs
    if "training_running" not in st.session_state:
        st.session_state.training_running= False
    if "training_completed" not in st.session_state:
        st.session_state.training_completed= False
    if "current_epoch" not in st.session_state:
        st.session_state.current_epoch=0
    if "training_history" not in st.session_state:
        st.session_state.training_history = []
    if "epoch_log_feed" not in st.session_state:
        st.session_state.epoch_log_feed = ""

    #Hyperparamter Sweeps
    if "sweep_history" not in st.session_state:
        st.session_state.sweep_history = []
    if "sweep_running" not in st.session_state:
        st.session_state.sweep_running=False

    #Export Status
    if "onnx_export_status" not in st.session_state:
        st.session_state.onnx_export_status = "NOT EXPORTED"
    if "export_logs" not in st.session_state:
        st.session_state.export_logs=""
    if "active_compilation_details" not in st.session_state:
        st.session_state.active_compilation_details= {}

    #LOGS DASHBOARD MANAGER

    if "system_logs_list" not in st.session_state:
        st.session_state.system_logs_list = AdvancedLoggerManager.generate_system_logs(seed=42)

    # INSTANTIATE FILE SYSTEM HELPER

    fs = VirtualFilesystem(st.session_state.drive_path)


main()

#SIDEBAR PERSISTENT CONTROLS & STAGE NAVIGATION CONTROLLER
with st.sidebar:
    st.markdown(
        """<div style='text-align': center; margin-bottom: 20px;'>
                <h2 style= 'margin:0; color: #00ffcc;'> Deep Learning Lab </h2>
                <p style='color': #6b7280; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em;>Platform Console v1.2.0</p>
            </div>
            """,
            unsafe_allow_html=True
    )

    st.divider()

    #Navigation controls based on current workflow status
    st.markdown("### WORKFLOW STAGES")


    stages = [
        ("Stage 1: System SMI Console", 1),
        ("Stage 2: Storage & Augmentation", 2),
        ("Stage 3: Architecture Playground", 3),
        ("Stage 4: AutoML Sweeps Matrix", 4),
        ("Stage 5: Neural Math Lab", 5),
        ("Stage 6: Roofline Benchmarks", 6),
        ("Stage 7: Convergence Staging", 7),
        ("Stage 8: Edge Compilers & FASR", 8),
        ("Stage 9: Weights Export Vault", 9)
    ]

    for name, idx in stages:
        disabled = False
        #Prevent users from entering training stage or export stages if procedures are not completed
        if idx > 1 and not st.session_state.hardware_locked:
            disabled = True
        if idx > 2 and not st.session_state.drive_mounted:
            disabled = True
        if idx > 6 and not st.session_state.uploaded_dataset_type and not st.session_state.active_dataset_preset :
            disabled = True
        if idx > 8 and not st.session_state.training_completed:
            disabled = True

        icon="Locked" if disabled else "Goo!"
        btn_label = f"{icon} {name}"

        #Highlight current stage visual button styling
        if idx == st.session_state.get("stage",0):
            btn_label = f"{name.upper()}"

        if st.button(btn_label, key=f"nav_btn_{idx}", disabled=disabled):
            st.session_state.stage = idx
            st.rerun()

    st.divider()

    st.markdown("### LIVE SYSTEM STATUS")
    st.markdown(f"**GPU**: `{st.session_state.selected_gpu}`")
    st.markdown(f"**Host**: `{st.session_state.system_provider.split('-')[0]}`")
    st.markdown(f"**Lock State**: `{'SECURE' if st.session_state.hardware_locked else 'UNBOUND'}`")
    st.markdown(f"**Mounted**: `{'YES' if st.session_state.drive_mounted else 'NO'}`")

    st.divider()

    #Simple Reset Handler
    if st.button("FULL SANDBOX WORKSPACE RESET"):
        st.session_state.clear()
        st.rerun()

#HEADER BADGES AREA: STATEFUL ISOLATION OVERVIEW
badge_runtime = "badge-active" if st.session_state.hardware_locked else "badge-idle"
badge_label = "RUNTIME: ACTIVE" if st.session_state.hardware_locked else "RUNTIME: INITIALIZING"

badge_training_state = "badge-idle"
badge_training_label =" TRAINING: IDLE"
if st.session_state.training_running:
    badge_training_state= "badge-training"
    badge_training_label= " TRAINING: CALCULATING"
elif st.session_state.training_completed:
    badge_training_state = "badge-active"
    badge_training_label = " TRAINING: COMPLETED"

st.markdown(
    f"""
    <div class='badge-container'>
    <div class='glow-badge {badge_runtime}'>{badge_label}</div>
    <div class='glow-badge {badge_training_state}'>{badge_training_label}</div>
    <div class='glow-badge badge-instance'> {st.session_state.system_provider.split('-')[0].upper()}</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Deep Learning Lifecycle Workspace")
st.markdown(
    "A unified training platform built for deep learning operations. Configure system resources, "
    "stage target filesystems, explore hyperparameters grids, and trace active neural conversion inside "
    "a sandbox styled like advanced AI research lab"
)
st.divider()


#STAGE 1: THE RUNTIME  INITILIZATION DASHBOARD
if st.session_state.stage == 1:
        st.header("Stage 1: Hardware Staging & System Handshake")
        st.markdown(
            "Before launching deep learning training iterations, you must initialize the host container hardware interface. "
            "Select an accelerator GPU profile matching your performance constraints, execute the resource handshake, "
            "and establish PCIe system mapping binds."
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'>Hardware Configuration</div>"
                "</div>",
                unsafe_allow_html=True
            )
            
            # Lock selections if hardware handshake is completed
            provider = st.selectbox(
                "Host Infrastructure Provider",
                options=LabConfigs.SYSTEM_PROVIDERS,
                index=LabConfigs.SYSTEM_PROVIDERS.index(st.session_state.system_provider),
                disabled=st.session_state.hardware_locked
            )
            st.session_state.system_provider = provider
            
            selected_gpu = st.selectbox(
                "Target GPU Core Profile",
                options=list(LabConfigs.GPU_PROFILES.keys()),
                index=list(LabConfigs.GPU_PROFILES.keys()).index(st.session_state.selected_gpu),
                disabled=st.session_state.hardware_locked
            )
            st.session_state.selected_gpu = selected_gpu
            
            # Displays hardware profile summary parameters
            spec = LabConfigs.GPU_PROFILES[selected_gpu]
            st.markdown(f"**Cuda Cores**: `{spec['cuda_cores']:,}` | **VRAM**: `{spec['vram_gb']} GB`")
            st.markdown(f"**Max Power**: `{spec['max_power']}` | **Thermal limit**: `{spec['max_temp']}`")
            st.markdown(f"**Estimated Compute Power**: `{spec['simulated_tflops']} TFLOPS (FP16)`")
            
            st.divider()
            
            if not st.session_state.hardware_locked:
                if st.button("INITIALIZE HARDWARE HANDSHAKE"):
                    try:
                        # Defensive UI status simulation blocks
                        with st.status("Querying System Resources...", expanded=True) as status:
                            st.write("Initializing system hardware handshakes...")
                            time.sleep(0.8)
                            st.write(f"Binding PCIe lanes targets to virtual driver ({spec['pcie_bandwidth']})...")
                            time.sleep(0.6)
                            st.write(f"Pre-allocating GPU memory registers and VRAM buffers ({spec['vram_gb']}GB)...")
                            time.sleep(0.7)
                            st.write("Generating system topologies tracing sheets...")
                            time.sleep(0.4)
                            status.update(label="Hardware Handshake Completed. Secure Connection Established!", state="complete")
                        
                        st.session_state.hardware_locked = True
                        st.session_state.nvidia_smi_log = HardwareTelemetryEngine.generate_nvidia_smi(selected_gpu, seed=random.randint(1, 100))
                        st.success(f"Hardware initialization succeeded! Target GPU: '{selected_gpu}' is bound to active PID process groups.")
                        time.sleep(1.0)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Defensive Exception Triggered during Handshake: {str(e)}")
            else:
                st.warning("Host Hardware Binds are Locked. Reset workspace elements to pick a different accelerator.")
                if st.button("RELEASE HARDWARE BINDS"):
                    st.session_state.hardware_locked = False
                    st.session_state.nvidia_smi_log = ""
                    st.rerun()
                    
        with col2:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'>Active CUDA System SMI Console</div>"
                "</div>",
                unsafe_allow_html=True
            )
            
            if st.session_state.hardware_locked:
                st.markdown(
                    "**Simulated Execution Frame: `!nvidia-smi`**"
                )
                st.code(st.session_state.nvidia_smi_log, language="bash")
                
                st.markdown("#### Mapped Device Topology Specs")
                topo = HardwareTelemetryEngine.get_system_topology_summary(st.session_state.selected_gpu)
                
                topo_df = pd.DataFrame(list(topo.items()), columns=["System Telemetry Parameter", "Reported Value"])
                st.table(topo_df)
            else:
                st.info("System core idle. Click 'Initialize Hardware Handshake' to query container SMI mappings.")
                
                # Interactive raw ASCII visual mock terminal display
                st.code(
                    """
                    $ !nvidia-smi
                    NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
                    Make sure that the latest NVIDIA driver is installed and running.
                    
                    Error: Hardware Handshake Uninitialized.
                    """, 
                    language="bash"
                )
                
        # Interactive custom shell console frame
        st.markdown("---")
        st.markdown("### Custom Debugging Command Console")
        st.markdown("Execute standard terminal commands inside the virtual sandbox system shell.")
        
        cmd_input = st.text_input("Console Input CLI", value="python3 -c 'import torch; print(torch.cuda.is_available())'", placeholder="Type command e.g., lscpu, uname -a...")
        if st.button("RUN COMMAND", key="run_terminal_command"):
            if not st.session_state.hardware_locked:
                st.error("No active system container detected. Handshake hardware first.")
            else:
                st.markdown("##### Virtual Console Output:")
                with st.spinner("Executing command..."):
                    time.sleep(0.5)
                    # Simulate terminal logs response
                    if "torch" in cmd_input:
                        cmd_res = "True\nCUDA version: 12.2\nDevice Name: " + st.session_state.selected_gpu
                    elif "uname" in cmd_input:
                        cmd_res = "Linux workspace-vnode-3212 5.15.0-101-generic #111-Ubuntu SMP x86_64 x86_64 GNU/Linux"
                    elif "lscpu" in cmd_input:
                        cmd_res = "Architecture:            x86_64\nCPU op-mode(s):        32-bit, 64-bit\nCore(s) per socket:      32\nThread(s) per core:     2"
                    else:
                        cmd_res = f"bash: {cmd_input.split()[0]}: command not found or sandbox restriction applied."
                    
                    st.code(f"$ {cmd_input}\n{cmd_res}", language="bash")

    # STAGE 2: STAGING AREA & DRIVE MOUNTING (DATA WORKSPACE CELL)
elif st.session_state.stage == 2:
        st.header("Stage 2: Storage Staging & Augmentation Sandbox")
        st.markdown(
            "Mount your training storage configurations (simulating Google Drive connection `/content/drive/MyDrive`) "
            "to check dataset files distributions, class annotations structures, and quality distribution."
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'>Drive Mounting Core</div>"
                "</div>",
                unsafe_allow_html=True
            )

            drive_input = st.text_input(
                "Virtual Drive Connection Path",
                value=st.session_state.drive_path,
                disabled=st.session_state.drive_mounted
            )
            st.session_state.drive_path = drive_input

            if not st.session_state.drive_mounted:
                if st.button("SECURELY MOUNT DRIVE VOLUME"):
                    with st.spinner("Connecting virtual drive directories..."):
                        time.sleep(1.0)
                    st.session_state.drive_mounted = True
                    st.success(f"Drive mounted successfully at `{st.session_state.drive_path}`! Mapped file tree system populated below.")
                    st.rerun()
            else:
                st.success(f"Drive Connected at `{st.session_state.drive_path}`")
                if st.button("DISCONNECT STORAGE VOLUME"):
                    st.session_state.drive_mounted = False
                    st.session_state.uploaded_dataset_type = None
                    st.session_state.dataset_filename = ""
                    st.session_state.classes_detected = []
                    st.session_state.active_dataset_preset = None
                    st.rerun()

            st.divider()

            st.markdown("### Dataset Uploader File Area")
            st.markdown("Upload dataset configuration files (`.yaml`/`.zip`) to populate the deep learning staging pipelines.")

            uploaded_file = st.file_uploader(
                "Dataset Config or Package Archive",
                type=["yaml", "zip"],
                disabled=not st.session_state.drive_mounted
            )

            if uploaded_file is not None:
                # Store information in session state
                st.session_state.dataset_filename = uploaded_file.name
                
                # Check extension types
                if uploaded_file.name.endswith(".yaml"):
                    try:
                        yaml_content = uploaded_file.read().decode("utf-8")
                        st.session_state.uploaded_dataset_type = "yaml"
                        st.session_state.dataset_yaml_data = yaml_content
                        st.session_state.active_dataset_preset = None
                        
                        # Validate structure
                        is_ok, parsed, msg = DatasetValidator.parse_yaml_content(yaml_content)
                        if is_ok and parsed:
                            st.session_state.classes_detected = list(parsed["names"].values())
                            st.toast("YAML Dataset config parsed successfully!")
                            
                            # Add mock file metadata to workspace
                            fs.add_virtual_file("configs", uploaded_file.name, f"{len(yaml_content)} B", "yaml", yaml_content)
                        else:
                            st.error(f"Dataset YAML Syntax Error: {msg}")
                    except Exception as e:
                        st.error(f"Error parsing dataset YAML: {str(e)}")

                elif uploaded_file.name.endswith(".zip"):
                    st.session_state.uploaded_dataset_type = "zip"
                    # ZIP analysis simulation
                    st.session_state.classes_detected = ["classA", "classB", "classC"] # default fallback
                    st.session_state.active_dataset_preset = None

                    fs.add_virtual_file("dataset", uploaded_file.name, f"{uploaded_file.size/1024:.2f} KB", "archive")
                    st.toast("Zip dataset archive staged to workspace!")

            st.markdown("Don't have a dataset? Choose a workspace preset:")
            selected_preset = st.selectbox(
                "Choose Staging Preset",
                options=list(LabConfigs.DATASET_PRESETS.keys()),
                disabled=not st.session_state.drive_mounted
            )

            if st.button("STAGE SELECTION CONFIG PRESET", disabled=not st.session_state.drive_mounted):
                preset = LabConfigs.DATASET_PRESETS[selected_preset]
                st.session_state.dataset_filename = selected_preset
                st.session_state.classes_detected = preset["classes"]
                st.session_state.active_dataset_preset = selected_preset

                if "content" in preset and isinstance(preset["content"], str) and not preset["content"].endswith(".zip"):
                    st.session_state.uploaded_dataset_type = "yaml"
                    st.session_state.dataset_yaml_data = preset["content"]
                    fs.add_virtual_file("configs", "preset_dataset.yaml", f"{len(preset['content'])} B", "yaml", preset["content"])
                else:
                    st.session_state.uploaded_dataset_type = "zip"
                    fs.add_virtual_file("datasets", "preset_dataset.zip", "4.2 MB", "archive")

                st.success(f"Staged preset '{selected_preset}' containing classes {preset['classes']}")

        with col2:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'>Workspace File System Mappings</div>"
                "</div>",
                unsafe_allow_html=True
            )

            if st.session_state.drive_mounted:
                st.markdown(f"**Root Mount: `{st.session_state.drive_path}`**")

                # Render current virtual files map inside visual console HTML frame
                tree_html = fs.render_tree_html(st.session_state.filesystem)
                st.markdown(f"<div class='console-frame'>{tree_html}</div>", unsafe_allow_html=True)

                # Active inspection card for uploaded configurations files
                if st.session_state.uploaded_dataset_type == "yaml":
                    st.markdown("### Mapped Class Names (.YAML)")
                    st.code(st.session_state.dataset_yaml_data, language="yaml")

                elif st.session_state.uploaded_dataset_type == "zip":
                    st.markdown("### Archive Properties Validator")

                    # Generate statistics metrics
                    stats = DatasetValidator.calculate_synthetic_zip_stats(st.session_state.classes_detected)

                    st.markdown("**Validation Splits Image Count**")
                    splits_df = pd.DataFrame(list(stats["Detected Images Split"].items()), columns=["Partition", "Image Count"])
                    st.table(splits_df)

                    st.markdown("**Classes Annotations Balance Metric**")
                    classes_df = pd.DataFrame.from_dict(stats["Mapped Annotations Logs"], orient='index')
                    st.dataframe(classes_df)
            else:
                st.info("Mount Storage Drive to view active project filesystems and validate data models.")

        # INTERACTIVE IMAGE AUGMENTATION SANDBOX
        st.markdown("---")
        st.markdown("### Interactive Dataset Image Augmentation Sandbox")
        st.markdown(
            "Test pixel-level augmentation dynamics on synthetic deep learning matrices. "
            "Adjust sliders to visualize how noise distributions, spatial blur filters, and intensity changes "
            "degrade and scale boundary coordinates maps."
        )
        
        if not st.session_state.drive_mounted:
            st.info("Connect the Virtual Storage Drive above to unlock the augmentation playground.")
        else:
            aug_col1, aug_col2 = st.columns([1, 2])
            
            with aug_col1:
                st.markdown(
                    "<div class='glass-card'>"
                    "<div class='glass-card-header'>Augmentation Strengths</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                noise_sl = st.slider("Gaussian Noise Variance", min_value=0.0, max_value=0.5, value=st.session_state.aug_noise, step=0.05)
                st.session_state.aug_noise = noise_sl
                
                blur_sl = st.slider("Blur Kernel Filter Scale", min_value=1, max_value=7, value=st.session_state.aug_blur, step=2)
                st.session_state.aug_blur = blur_sl
                
                contrast_sl = st.slider("Intensity Contrast Factor", min_value=0.2, max_value=2.0, value=st.session_state.aug_contrast, step=0.1)
                st.session_state.aug_contrast = contrast_sl
                
                st.divider()
                
                st.markdown(
                    "#### Simulated bounding coordinates target box:\n"
                    "- Class Tag: `detectable_object`\n"
                    "- Anchors map: `[xmin: 40, ymin: 30, xmax: 80, ymax: 70]`"
                )
                
            with aug_col2:
                st.markdown(
                    "<div class='glass-card'>"
                    "<div class='glass-card-header'>Visualized Bounding Matrix Output</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                if plt is not None and np is not None:
                    try:
                        # 1. Generate base canvas matrix
                        raw_canvas = ImageAugmentationSimulator.generate_mock_image(size=120)
                        
                        # 2. Apply augmentation operators
                        noised = ImageAugmentationSimulator.apply_noise(raw_canvas, st.session_state.aug_noise)
                        blurred = ImageAugmentationSimulator.apply_blur(noised, st.session_state.aug_blur)
                        final_canvas = ImageAugmentationSimulator.apply_contrast(blurred, st.session_state.aug_contrast)
                        
                        # 3. Plot matrix using matplotlib
                        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0c0f16")
                        ax.set_facecolor("#080a0f")
                        
                        ax.imshow(final_canvas)
                        
                        # Add red border representing coordinates box
                        rect = patches.Rectangle(
                            (40, 30), 40, 40, 
                            linewidth=2, 
                            edgecolor="#ef4444", 
                            facecolor="none",
                            linestyle="--"
                        )
                        ax.add_patch(rect)
                        
                        ax.text(
                            42, 25, "detectable_object (94%)", 
                            color="#ef4444", 
                            fontweight="bold",
                            fontsize=9,
                            bbox=dict(facecolor='#080a0f', alpha=0.8, edgecolor='none', pad=2)
                        )
                        
                        ax.set_title("Processed Matrix Target Augmentation", color="#ffffff", fontweight="bold")
                        ax.axis("off")
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error drawing canvas: {str(e)}")
                else:
                    st.info("Augmentation graphs disabled because NumPy/Matplotlib packages are uninstalled.")

#Stage 3: Neural Architecture Playground
elif st.session_state.stage == 3:
    st.header("Stage 3: Neural Backbone Architecture Playground")
    st.markdown(
        "Graphically construct your deep learning feature extraction backbones"
        "Append custom covolution, normalizations, and activation pooling layers to visually map"
        "layer dimensions and evaluates trainable parameters footprints"
    )

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown(
            "<div class='glass-card'>"
            "<div class='glass-card-header'>Add Custom Layer Definition</div>"
            "</div>",
            unsafe_allow_html=True
        )

        new_type = st.selectbox(
            "Layer Operation Type",
            options=["Conv2d", "BatchNorm2d", "SiLU", "MaxPool2d", "Concatenate"]
        )

        out_dim= st.number_input("Output Channels Width", min_value=16, max_value=1024, value=128, step=16)
        k_size= st.selectbox("Kernel Size Spatial Filter", options=[1,3,5,7], index=1)
        stride_size= st.selectbox("Stride Convolution Step", options=[1,2], index=0)

        if st.button("APPEND NEW LAYER COMPONENT"):
            next_id=len(st.session_state.custom_backbone_layers)+1

            #Format options mappings
            out_str = out_dim if new_type in ["Conv2d","BatchNorm2d"] else "-"
            k_str = k_size if new_type in ["Conv2d","MaxPool2d"] else "-"
            st_str = stride_size if new_type in ["Conv2d", "MaxPool2d"] else "-"

            st.session_state.custom_backbone_layers.append({
                "layer_id":next_id,
                "type":new_type,
                "out_channels":out_str,
                "kernel_size":k_str,
                "stride": st_str
            })
            st.success("New Layer added successfully")
            st.rerun()

        st.divider()

        st.markdown("### Backbone Templates Core")
        st.markdown("Load pre-configured core configurations maps representing classical ML architectures.")

        if st.button("RESET TO  RESNET-18 BACKBONE TEMPLATE"):
            st.session_state.custom_backbone_layers = [
                {"layer_id":1,"type":"Conv2d", "out_channels":64, "kernel_size":7, "stride":2},
                {"layer_id":2,"type":"BatchNorm2d", "out_channels":64, "kernel_size":"-", "stride":"-"},
                {"layer_id":3,"type":"SiLU","out_channels":"-","kernel_size":"-", "stride":"-"},
                {"layer_id":4,"type":"MaxPool2d","out_channels":"-","kernel_size":3, "stride":2},
                {"layer_id":5,"type":"Conv2d", "out_channels":64,"kernel_size":3, "stride":1},
                {"layer_id":6,"type":"Conv2d", "out_channels":128, "kernel_size":3, "stride":2}
            ]
            st.success("Successfully loaded ResNet-12 extraction channels")
            st.rerun()

        if st.button("RESET TO YOLOv8-NANO ACCELERATED BACKBONE"):
            st.session_state.custom_backbone_layers = [
                {"layer_id":1, "type":"Conv2d", "out_channels": 16, "kernel_size":3, "stride":2},
                {"layer_id":2, "type":"Conv2d","out_channels":32, "kernel_size":3, "stride":2},
                {"layer_id":3, "type":"BatchNorm2d", "out_channels":32, "kernel_size":"-","stride":"-"},
                {"layer_id":4, "type":"SiLU", "out_channels":"-","kernel_size":"-","stride":"-"},
                {"layer_id":5, "type":"Conv2d", "out_channels":64, "kernel_size":3, "stride":2}
            ]
            st.success("Successfully loaded YOLOv8 structural extraction templates")
            st.rerun()

    with col2:
        st.markdown(
            "<div class='glass-card'>"
            "<div class='glass-card-header'>Mapped Layers Dimensions & Graphs </div>"
            "</div>",
            unsafe_allow_html=True
        )

        #Print list of layers
        if st.session_state.custom_backbone_layers:
            layers_df=pd.DataFrame(st.session_state.custom_backbone_layers).astype(str)
            st.table(layers_df)

            #Estimated FLOPs calculations
            total_params = 0
            for layer in st.session_state.custom_backbone_layers:
                if layer["type"] == "Conv2d" and isinstance(layer["out_channels"], int):
                    total_params += (layer["out_channels"] * 9 * (layer["kernel_size"] if isinstance(layer["kernel_size"],int)else 3)**2)


            st.markdown(f"**Total Estimated Trainable Parameters**: `{total_params:,} Weight Coefficients`")
            st.markdown(f"**GigaFLOPs Footprint Estimation**: `{total_params* 0.0035:.2f} GFLOPs (Input size: 640x640)`")


            #Custom Code Generator Block
            st.markdown("### Mapped YOLOv8-style YAML Graph Configuration")

            yaml_code = "# YOLOv8 Custom Backbone Graph\nbackbone\n"
            for idx, layer in enumerate(st.session_state.custom_backbone_layers):
                yaml_code += f" -[{idx}, 1, {layer['type']}, [{layer.get('out_channels','-')}, {layer.get('kernel_size','-')}, {layer.get('stride','-')}]]\n"
            
            st.code(yaml_code, language="yaml")

            #Dynamic PyTorch Code Generator Tab
            st.markdown("### Generated PyTorch Operational Module Code")
            py_tabs = st.tabs(["Conv Class Module", "C2f Block Module"])
            with py_tabs[0]:
                st.code(YoloPyTorchCodeGenerator.generate_conv_module(), language="python")
            with py_tabs[1]:
                st.code(YoloPyTorchCodeGenerator.generate_c2f_module(), language="python")

            
            #Check for storage binds and save
            if st.button("SAVE BACKBONE TO CONFIG DIRECTORY"):
                fs.add_virtual_file("configs","custom_backbone.yaml",f"{len(yaml_code)} B","yaml",yaml_code)
                st.success("Custom_backbone.yaml successfully saved to Virtual Filesystems")
            else:
                st.info("Appended custom layer modules above to visualize feature graphs")


#Stage 4: HyperParamtere Tuning Sweeps (AutoML SANDBOX)
elif st.session_state.stage == 4:
        st.header(" Stage 4: AutoML Sandbox & Hyperparameter Grid Sweeps")
        st.markdown(
            "Optimize neural training trajectories before executing long-running runs. "
            "Simulate hyperparameter sweep distributions to trace Pareto accuracy frontiers and analyze "
            "correlations between batch scales, optimizer models, and learning rate targets."
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'> Sweep Search Space</div>"
                "</div>",
                unsafe_allow_html=True
            )
            
            sweep_backbone = st.selectbox(
                "Sweep Target Model",
                options=list(LabConfigs.YOLO_BACKBONES.keys()),
                index=1
            )
            
            sweep_runs_count = st.slider("Sweep Iterations Counts", min_value=5, max_value=30, value=12)
            
            sweep_optimizer = st.multiselect(
                "Optimizer Candidates Space",
                options=["AdamW", "SGD", "RMSprop"],
                default=["AdamW", "SGD"]
            )
            
            st.divider()
            
            if st.button(" EXECUTE AUTOMATED SWEEP SEARCH", disabled=st.session_state.sweep_running):
                st.session_state.sweep_running = True
                st.session_state.sweep_history = []
                
                with st.status("Executing AutoML Sweeps iterations...", expanded=True) as status:
                    for i in range(1, sweep_runs_count + 1):
                        st.write(f"Calculating Sweep Step Run {i}/{sweep_runs_count}...")
                        time.sleep(0.3)
                        
                        # Generate random sweep data parameters
                        run_spec = HyperparameterSweeper.generate_random_sweep(
                            run_id=i,
                            backbone=sweep_backbone,
                            seed=random.randint(1, 1000)
                        )
                        st.session_state.sweep_history.append(run_spec)
                        
                    status.update(label="Hyperparameter Sweep Runs calculations completed!", state="complete")
                    
                st.session_state.sweep_running = False
                st.success("Successfully traced accuracy frontier correlations matrix!")
                
        with col2:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'> Parallel Sweeps Telemetry Metrics</div>"
                "</div>",
                unsafe_allow_html=True
            )
            
            if st.session_state.sweep_history:
                sweep_df = pd.DataFrame(st.session_state.sweep_history)
                st.dataframe(sweep_df)
                
                # Check matplotlib to generate beautiful plot charts
                if plt is not None and pd is not None:
                    try:
                        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0c0f16")
                        ax.set_facecolor("#1a1f2c")
                        
                        # Filter completed runs
                        completed_df = sweep_df[sweep_df["Status"] == "COMPLETED"]
                        
                        # Plot learning rate vs Accuracy (mAP50) sized by Batch scale
                        if not completed_df.empty:
                            sc = ax.scatter(
                                completed_df["LR"],
                                completed_df["mAP50"],
                                s=completed_df["Batch Size"] * 5,
                                c=completed_df["Final Loss"],
                                cmap="viridis",
                                alpha=0.85,
                                edgecolors="white",
                                linewidths=0.5
                            )
                            cbar = fig.colorbar(sc, ax=ax)
                            cbar.set_label("Final Target Loss", color="#d1d5db")
                            cbar.ax.yaxis.set_tick_params(color="#d1d5db")
                            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#d1d5db")
                            
                            ax.set_title("Sweep Pareto Frontier: Learning Rate vs accuracy (mAP50)", color="#ffffff", fontsize=11, fontweight="bold")
                            ax.set_xlabel("Learning Rate Base (Log Scale)", color="#d1d5db")
                            ax.set_ylabel("Peak Accuracy Metric (mAP50)", color="#d1d5db")
                            ax.set_xscale("log")
                            ax.tick_params(colors="#d1d5db")
                            ax.grid(True, linestyle="--", alpha=0.1)
                            
                            st.pyplot(fig)
                        else:
                            st.warning("All sweep calculations resulted in diverged gradients loss. Scale back LR configurations.")
                    except Exception as e:
                        st.error(f"Error plotting sweep diagrams: {str(e)}")
                else:
                    st.info("Visual charts disabled because Matplotlib/Pandas features are uninstalled.")
                    
                # Identify best run parameters configurations
                best_run = max(st.session_state.sweep_history, key=lambda x: x["mAP50"])
                
                st.markdown("###  Top Performing Sweep Config")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Best Learning Rate", f"{best_run['LR']}", help="Optimal learning speed parameter")
                col_b.metric("Batch size & Optimizer", f"{best_run['Batch Size']} ({best_run['Optimizer']})")
                col_c.metric("Highest accuracy (mAP50)", f"{best_run['mAP50'] * 100:.2f}%")
                
                if st.button("⚙️ INJECT CHAMPION SWEEP HYPERPARAMETERS TO TRAINING MODULE"):
                    st.session_state.model_backbone = best_run["Backbone"]
                    st.session_state.batch_size = best_run["Batch Size"]
                    st.session_state.learning_rate = best_run["LR"]
                    st.session_state.optimizer_type = best_run["Optimizer"]
                    st.success("Tuned values successfully loaded to Stage 7 active training cell!")
            else:
                st.info(" Complete Stage 4 sweep iterations calculations to view automated hyperparameter graphs.")

#NEURAL MATHEMATICS & PURE PYTHON ALGORITHMS
elif st.session_state.stage == 5:
        st.header(" Stage 5: Neural Mathematics & Pure-Python Lab")
        st.markdown(
            "Study the computational models and foundational algorithms behind deep object detection. "
            "Interact with fully operational, pure-Python modules for Non-Maximum Suppression (NMS), "
            "Intersection-over-Union (IoU) matrices, and statistical confusion arrays."
        )
        
        math_tabs = st.tabs([" IoU & NMS Decoders", " Confusion Matrix Calculations", " Optimizer Mechanics Theory", " Pure-Python Weight Optimizer Staging", " Transformers Self-Attention Simulator"])
        
        with math_tabs[0]:
            st.markdown(
                "###  Bounding Box Overlaps & NMS Filtering"
            )
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(
                    "<div class='glass-card'>"
                    "<div class='glass-card-header'> Custom Box Overlap Coordinates</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                st.markdown("**Box 1 Coordinates (Format: [xmin, ymin, xmax, ymax])**")
                b1_x = st.slider("Box 1: xmin Anchor", 0, 100, 20)
                b1_y = st.slider("Box 1: ymin Anchor", 0, 100, 20)
                box1 = [b1_x, b1_y, b1_x + 40, b1_y + 40]
                st.markdown(f"Current Box 1: `{box1}`")
                
                st.divider()
                
                st.markdown("**Box 2 Coordinates**")
                b2_x = st.slider("Box 2: xmin Anchor", 0, 100, 35)
                b2_y = st.slider("Box 2: ymin Anchor", 0, 100, 35)
                box2 = [b2_x, b2_y, b2_x + 35, b2_y + 35]
                st.markdown(f"Current Box 2: `{box2}`")
                
            with col2:
                st.markdown(
                    "<div class='glass-card'>"
                    "<div class='glass-card-header'> Measured Mathematical Output</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                # Compute IoU and variants
                calculated_iou = NeuralMathPlayground.calculate_iou(box1, box2)
                calculated_giou = NeuralMathPlayground.calculate_giou(box1, box2)
                calculated_diou = NeuralMathPlayground.calculate_diou(box1, box2)
                
                st.metric("Measured Box Intersection-over-Union (IoU)", f"{calculated_iou * 100:.2f}%", help="Intersection over Union Area Ratio")
                st.metric("Measured Generalized IoU (GIoU)", f"{calculated_giou:.4f}", help="Handles non-overlapping distance penalty")
                st.metric("Measured Distance IoU (DIoU)", f"{calculated_diou:.4f}", help="Aligns box center distances")
                
                if plt is not None:
                    try:
                        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0c0f16")
                        ax.set_facecolor("#1a1f2c")
                        
                        r1 = patches.Rectangle((box1[0], box1[1]), box1[2]-box1[0], box1[3]-box1[1], linewidth=2, edgecolor="#06b6d4", facecolor="none", label="Box 1")
                        r2 = patches.Rectangle((box2[0], box2[1]), box2[2]-box2[0], box2[3]-box2[1], linewidth=2, edgecolor="#f59e0b", facecolor="none", label="Box 2")
                        
                        ax.add_patch(r1)
                        ax.add_patch(r2)
                        
                        ax.set_xlim(0, 150)
                        ax.set_ylim(0, 150)
                        ax.invert_yaxis()
                        ax.tick_params(colors="#d1d5db")
                        ax.legend()
                        ax.set_title(f"Visualized Intersection Matrix (IoU: {calculated_iou * 100:.1f}%)", color="#ffffff", fontweight="bold")
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error plotting box layouts: {str(e)}")
                        
            # Dynamic NMS Playground
            st.markdown("---")
            st.markdown("####  Non-Maximum Suppression (NMS) Processing Simulator")
            nms_threshold = st.slider("Overlap NMS IoU Threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
            
            candidate_boxes = [
                [30, 30, 70, 70],
                [32, 28, 71, 72],
                [35, 34, 68, 69],
                [80, 80, 110, 110]
            ]
            candidate_scores = [0.92, 0.78, 0.64, 0.85]
            
            st.markdown("**Staged candidate predictions (BBox coordinates + scores):**")
            preds_data = []
            for i in range(len(candidate_scores)):
                preds_data.append({"Prediction index": i, "Coordinates": str(candidate_boxes[i]), "Confidence Score": candidate_scores[i]})
            st.table(preds_data)
            
            if st.button(" RUN NON-MAXIMUM SUPPRESSION ALGORITHM"):
                final_keep = NeuralMathPlayground.execute_nms(candidate_boxes, candidate_scores, nms_threshold)
                st.success(f"NMS run completed! Kept box indices: `{final_keep}` (Collapsed duplicate inputs: `{list(set(range(len(candidate_scores))) - set(final_keep))}`)")
                
        with math_tabs[1]:
            st.markdown("###  Confusion Matrix Heatmaps & Evaluation Curves")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(
                    "<div class='glass-card'>"
                    "<div class='glass-card-header'>Target Mapped Classes Matrix</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                active_classes = st.session_state.classes_detected if st.session_state.classes_detected else ["class1", "class2", "class3"]
                
                conf_df = NeuralMathPlayground.compute_confusion_matrix(active_classes)
                st.markdown("**Numerical Confusion Matrix Counts**")
                st.dataframe(conf_df)
                
                if plt is not None:
                    try:
                        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0c0f16")
                        ax.set_facecolor("#1a1f2c")
                        
                        cax = ax.matshow(conf_df.values, cmap="Blues")
                        fig.colorbar(cax, ax=ax)
                        
                        ax.set_xticks(range(len(conf_df.columns)))
                        ax.set_yticks(range(len(conf_df.index)))
                        ax.set_xticklabels(list(conf_df.columns), color="#d1d5db")
                        ax.set_yticklabels(list(conf_df.index), color="#d1d5db")
                        
                        ax.tick_params(colors="#d1d5db")
                        ax.set_title("Visual Confusion Matrix Heatmap", color="#ffffff", fontweight="bold", pad=20)
                        
                        for i in range(len(conf_df.index)):
                            for j in range(len(conf_df.columns)):
                                ax.text(j, i, str(conf_df.values[i, j]), va='center', ha='center', color="white" if conf_df.values[i, j] > 100 else "black")
                                
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error compiling confusion heatmap: {str(e)}")
                        
            with col2:
                st.markdown(
                    "<div class='glass-card'>"
                    "<div class='glass-card-header'> Precision-Recall Precision Limits</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                recalls, precisions = NeuralMathPlayground.calculate_precision_recall_points()
                
                if plt is not None:
                    try:
                        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0c0f16")
                        ax.set_facecolor("#1a1f2c")
                        
                        ax.plot(recalls, precisions, label="Model YOLOv8 Target", color="#00ffcc", linewidth=2.5)
                        ax.fill_between(recalls, precisions, color="#00ffcc", alpha=0.1)
                        
                        ax.set_xlim(0, 1.05)
                        ax.set_ylim(0, 1.05)
                        ax.set_xlabel("Recall Rate", color="#d1d5db")
                        ax.set_ylabel("Precision Rate", color="#d1d5db")
                        ax.tick_params(colors="#d1d5db")
                        ax.grid(True, linestyle="--", alpha=0.1)
                        ax.legend()
                        ax.set_title("Precision-Recall Curve (AUC: 0.88)", color="#ffffff", fontweight="bold")
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error plotting evaluation curve: {str(e)}")
                        
        with math_tabs[2]:
            st.markdown("###  Dynamic Optimizer Calculations & Momentum Mechanics")
            opt_select = st.selectbox("Select Optimization Algorithm Target", options=list(OptimizerTheoryMatrix.DETAILS.keys()))
            opt_details = OptimizerTheoryMatrix.DETAILS[opt_select]
            
            col_th1, col_th2 = st.columns([1, 1])
            with col_th1:
                st.markdown(f"#### {opt_details['title']}")
                st.markdown(opt_details["description"])
                
                st.markdown("**Core Hyperparameter Bounds**")
                params_df = pd.DataFrame(list(opt_details["parameters"].items()), columns=["Hyperparameter Option", "Typical Bound"])
                st.table(params_df)
                
            with col_th2:
                st.markdown("#### Mathematical Formulation equations:")
                st.latex(opt_details["math"])
                
                st.divider()
                st.markdown("####  Dynamic Loss Functions Physics Matrix")
                fl_details = AdvancedDeepLearningTheory.get_focal_loss_explanation()
                st.markdown(f"**{fl_details['title']}**")
                st.latex(fl_details["equation"])
                st.markdown(fl_details["description"])

        with math_tabs[3]:
            st.markdown("###  Pure-Python Weight Optimizer Staging sandbox")
            st.markdown(
                "Interact with a functional deep learning weight matrix initialized under standard statistical rules. "
                "Execute single SGD optimization steps, calculating gradients decay and plotting live "
                "spatial weight distribution maps in real-time."
            )
            
            opt_col1, opt_col2 = st.columns([1, 2])
            
            with opt_col1:
                st.markdown(
                    "<div class='glass-card'>"
                    "<div class='glass-card-header'> Initialization Parameters</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                init_method = st.selectbox("Weight Distribution Profile", options=["Xavier Normal", "He Normal", "Standard Normal"])
                
                if st.button(" RE-INITIALIZE WEIGHT TENSOR"):
                    st.session_state.visual_weights_matrix = NeuralLayerOptimizerSimulator.initialize_layer_weights(dim=16, init_type=init_method)
                    st.session_state.visual_weights_velocity = np.zeros((16, 16))
                    st.session_state.layer_init_method = init_method
                    st.success(f"Weights tensor re-initialized under {init_method} bounds!")
                    st.rerun()
                    
                st.divider()
                
                st.markdown("#### Staging SGD optimization parameters:")
                sgd_lr = st.slider("Optimization Learning Rate (eta)", 0.001, 0.2, 0.05, step=0.01)
                sgd_mom = st.slider("Momentum Dampening (gamma)", 0.0, 0.99, 0.9, step=0.05)
                sgd_wd = st.slider("Decoupled L2 Weight Decay (lambda)", 0.0, 0.01, 0.0005, step=0.0005, format="%.5f")
                
                if st.button("⚡ EXECUTE SINGLE GRADIENT OPTIMIZATION STEP"):
                    np.random.seed(random.randint(1, 1000))
                    dummy_gradients = np.random.normal(0, 0.05, (16, 16))
                    
                    new_w, new_v = NeuralLayerOptimizerSimulator.step_matrix_optimization(
                        weights=st.session_state.visual_weights_matrix,
                        gradients=dummy_gradients,
                        velocity=st.session_state.visual_weights_velocity,
                        learning_rate=sgd_lr,
                        momentum=sgd_mom,
                        weight_decay=sgd_wd
                    )
                    st.session_state.visual_weights_matrix = new_w
                    st.session_state.visual_weights_velocity = new_v
                    st.success("Gradient descent step completed! Matrix values updated.")
                    st.rerun()
                    
            with opt_col2:
                st.markdown(
                    "<div class='glass-card'>"
                    "<div class='glass-card-header'> Spatial Weights Heatmap & Histogram</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                if plt is not None and np is not None:
                    try:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="#0c0f16")
                        ax1.set_facecolor("#1a1f2c")
                        ax2.set_facecolor("#1a1f2c")
                        
                        im = ax1.imshow(st.session_state.visual_weights_matrix, cmap="RdBu", aspect="equal")
                        fig.colorbar(im, ax=ax1)
                        ax1.set_title("Layer weights Heatmap [16 x 16]", color="#ffffff", fontweight="bold")
                        ax1.axis("off")
                        
                        ax2.hist(st.session_state.visual_weights_matrix.flatten(), bins=20, color="#06b6d4", edgecolor="black", alpha=0.85)
                        ax2.set_title("Weights Numerical Distribution", color="#ffffff", fontweight="bold")
                        ax2.set_xlabel("Weight Value", color="#d1d5db")
                        ax2.set_ylabel("Frequencies Count", color="#d1d5db")
                        ax2.tick_params(colors="#d1d5db")
                        ax2.grid(True, linestyle="--", alpha=0.1)
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error compiling weights optimization diagrams: {str(e)}")

        with math_tabs[4]:
            st.markdown("###  Transformers Self-Attention Simulator Sandbox")
            st.markdown(
                "Visualize dynamic Multi-Head Self-Attention layers score matrices calculations. "
                "Simulate Query-Key dot products scaling and Softmax projections patterns interactively."
            )
            
            att_col1, att_col2 = st.columns([1, 2])
            
            with att_col1:
                st.markdown(
                    "<div class='glass-card'>"
                    "<div class='glass-card-header'> Attention Matrix Config</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                seq_len = st.slider("Sequence Length (Token count)", min_value=8, max_value=32, value=16, step=4)
                
                if st.button("🔮 GENERATE RANDOM ATTENTION RUN"):
                    st.toast("Self-Attention Map projected successfully!")
                    
            with att_col2:
                st.markdown(
                    "<div class='glass-card'>"
                    "<div class='glass-card-header'> Softmax Self-Attention Heatmap</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                if plt is not None and np is not None:
                    try:
                        att_map = SelfAttentionSimulator.compute_attention_map(seq_len=seq_len)
                        
                        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0c0f16")
                        ax.set_facecolor("#1a1f2c")
                        
                        im = ax.imshow(att_map, cmap="viridis", aspect="equal")
                        fig.colorbar(im, ax=ax)
                        
                        ax.set_title(f"Scaled Dot-Product Attention [Seq: {seq_len} x {seq_len}]", color="#ffffff", fontweight="bold")
                        ax.set_xlabel("Key Projection Positions", color="#d1d5db")
                        ax.set_ylabel("Query Projection Positions", color="#d1d5db")
                        ax.tick_params(colors="#d1d5db")
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error drawing self-attention map: {str(e)}")

    #ROOFLINE BENCHMARKS
elif st.session_state.stage == 6 :
    st.header("Stage 6: Roofline Performance & STREAM Benchmarks")
    st.markdown(
        "Roofline model analysis maps deep learning operational intensity against hardware limits"
        "Evaluates if target backbones are memory bandwidth bound or compute bound under select"
        "accelerator profiles"

    )

    bench_tabs = st.tabs(["Dynamic Roofline Bounds", "STREAM Memory Bandwidth Sweeps"])

    with bench_tabs[0]:
        col1, col2 = st.columns([1,1])
        
        with col1:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'>Staging Hardware Constraints</div>"
                "</div>",
                unsafe_allow_html=True
            )

            benchmark_gpu = st.selectbox(
                "Benchmark Target Acceleration GPU",
                options=list(LabConfigs.GPU_PROFILES.keys()),
                index=list(LabConfigs.GPU_PROFILES.keys()).index(st.session_state.selected_gpu)
            )

            benchmark_flops = st.number_input(
                "Staging Architecture FLOPs (GigaFLOPs)",
                min_value=1.0,
                max_value=500.0,
                value=float(LabConfigs.YOLO_BACKBONES[st.session_state.model_backbone]["flops"].replace("B", "")),
                step=1.0
            )

            st.divider()

            stats= RooflineBenchmarkEngine.get_roofline_profile(benchmark_gpu, benchmark_flops)

            st.markdown("#### Hardware Benchmark Profile Metrics")
            st.markdown(f"**Peak Compute Performance**: `{stats['Peak FLOPS Target']}`")
            st.markdown(f"**Peak Memory Badwidth**: `{stats['Peak Memory Bandwidth']}`")
            st.markdown(f"**Operational intensity Threshold (Knee Point)**: `{stats['Hardware Knee Point']}`")
            st.markdown(f"**Calculated Model Intensity**: `{stats['Calculated Model Intensity']}`")

            st.divider()

            st.markdown(f"### Target System Bottleneck: **{stats['System Bottleneck Status']}**")
            st.metric("Max Achievable Performance", stats["Max Achievable Performance"])

        with col2:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'> Target Roofline Model diagram </div>"
                "</div>",
                unsafe_allow_html=True
            )

            if plt is not None:
                try:
                    spec= LabConfigs.GPU_PROFILES[benchmark_gpu]
                    peak_flops= spec["simulated_tflops"]
                    peak_bw = spec.get("peak_bandwidth_gbs", 2039.0)

                    fig, ax= plt.subplots(figsize=(7,5), facecolor="#0c0f16")
                    ax.set_facecolor("#1a1f2c")

                    intensitites = np.logspace(-2,4,200)
                    achievable_flops = np.minimum(peak_flops, (intensitites*peak_bw)/1000.0)

                    ax.plot(intensitites, achievable_flops, label=f"Roofline Bound ({benchmark_gpu})", color="#e11d48", linewidth=2.5)

                    model_intensity = RooflineBenchmarkEngine.calculate_operational_intensity(benchmark_flops* 1e9, 22.4 * 1e6)
                    model_achievable = min(peak_flops, (model_intensity * peak_bw) / 1000.0)

                    ax.scatter(model_intensity, model_achievable, color="#00ffcc", s=150, zorder=5, label=f"Model Backbone Point")
                    ax.text(model_intensity * 1.5, model_achievable, "Backbone", color="#00ffcc", fontweight="bold", fontsize=10 )

                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel("Operational Intensity (FLOPs/Byte)", color="#d1d5db")
                    ax.set_ylabel("Performance Target (TFLOPS)", color="#d1d5db")
                    ax.tick_params(colors="#d1d5db")
                    ax.grid(True, which="both", linestyle="--", alpha=0.1)
                    ax.legend()
                    ax.set_title(f"Visualized Roofline Model Performance Limits", color="#ffffff", fontweight="bold")
                    

                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error plotting Roofline models: {str(e)}")

    with bench_tabs[1]:
        st.markdown("### STREAM Memory Bandwidth Sweeps Dashboard")
        st.markdown(
            "Run simulated dynamic memory transfers (Copy, Scale, Add, Triad) representing STREAM benchmark configurations"
            "Compare calculated memory performance scaling profiles against selected device bus limits"
        )

        stream_col1, stream_col2 = st.columns([1,2])
        with stream_col1:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'>Sweep Constraints</div>"
                "</div>",
                unsafe_allow_html=True
            )

            target_sweep_gpu = st.selectbox("STREAM Mapped GPU Target", options=list(LabConfigs.GPU_PROFILES.keys()), key="stream_gpu_select")


            if st.button("EXECUTE STREAM BENCHMARK SWEEP"):
                st.toast("STREAM memory sweep calculations completed")

        with stream_col2:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-class-header'>STREAM Benchmark Performance</div>"
                "</div>",
                unsafe_allow_html=True
            )

            stream_bw = StreamMemorySimulator.get_stream_bandwidths(target_sweep_gpu)
            stream_df = pd.DataFrame(list(stream_bw.items()), columns=["STREAM Operation Kernel", "Measured Bandwidth (GB/s)"])
            st.table(stream_df)

            if plt is not None:
                try:
                    fig, ax = plt.subplots(figsize=(6,4), facecolor="#0c0f16")
                    ax.set_facecolor("#1a1f2c")

                    ax.bar(stream_bw.keys(), stream_bw.values(), color="#3b82f6", edgecolor="black", alpha=0.85)
                    ax.set_ylabel("Meausred Transfer Bandwidth (GB/s)", color="#d1d5db")
                    ax.tick_params(colors="#d1d5db")
                    ax.grid(True, linestyle="--", alpha=0.1)
                    ax.set_title(f"STREAM Memory Bandwidth Sweep ({target_sweep_gpu})", color="#ffffff", fontweight="bold")


                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error drawing STREAM graphs: {str(e)}")

    #DEEP LEARNING MODEL TRAINING
elif False:
    st.header("Stage 6: Roofline Performance & STREAM Benchmarks")
    st.markdown(
        "Roofline model analysis maps deep learning operational intensity against hardware limits"
        "Evaluate if target backbones are memory bandwidth bound or compute bound under select"
        "accelerator profiles"
    )

    bench_tabs = st.tabs(["Dynamic Roofline Bounds", "STREAM MEMORY Bandwidth Sweeps"])

    with bench_tabs[0]:
        col1, col2= st.columns([1,1])

        with col1:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'>Staging Hardware Constraints</div>"
                "</div>",
                unsafe_allow_html=True
            )

            benchmark_gpu = st.selectbox(
                "Benchmark Target Acceleration GPU",
                options=list(LabConfigs.GPU_PROFILES.keys()),
                index=list(LabConfigs.GPU_PROFILES.keys()).index(st.session_state.selected_gpu)
            )

            benchmark_flops = st.number_input(
                "Staging Architecture FLOPs (GigaFLOPs)",
                min_value=1.0,
                max_value=500.0,
                value=float(LabConfigs.YOLO_BACKBONES[st.session_state.model_backbone]["flops"].replace("B","")),
                step=1.0
            )

            st.divider()

            stats = RooflineBenchmarkEngine.get_roofline_profile(benchmark_gpu, benchmark_flops)

            st.markdown("#### Hardware Benchmark Profile Metrics")
            st.markdown(f"**Peak Compute Performance: `{stats['Peak FLOPS Target']}`")
            st.markdown(f"**Peak Memory Bandwidth: `{stats['Peak Memory Bandwidth']}`")
            st.markdown(f"**Operational intensity Threshold (Knee Point)**: `{stats['Hardware Knee Point']}`")
            st.markdown(f"**Calculated Model Intensity**:`{stats['Caclulated Model Intensity']}`")

            st.divider()

            st.markdown(f"### Target System Bottleneck: **{stats['System Bottleneck Status']}")
            st.markdown("Max Achievable Performance", stats["Max Achievable Performance"])

        with col2:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'>Target Roofline Model Diagram</div>"
                "</div>",
                unsafe_allow_html=True
            )

            if plt is not None:
                try:
                    spec = LabConfigs.GPU_PROFILES[benchmark_gpu]
                    peak_flops = spec["simulated_tflops"]
                    peak_bw= spec.get("peak_bandwidth_gbs", 2039.0)

                    fig, ax = plt.subplots(figsize=(7,5), facecolor="#0c0f16")
                    ax.set_facecolor("#1a1f2c")

                    intensitites = np.logspace(-2,4,200)
                    achievable_flops= np.minimum(peak_flops, (intensitites * peak_bw) / 1000.0)

                    ax.plot(intensitites, achievable_flops, label=f"Roofline Bound ({benchmark_gpu})", color="#e11d48", linewidth=2.5)


                    model_intensity= RooflineBenchmarkEngine.calculate_operational_intensity(benchmark_flops*1e9,22.4*1e6)
                    model_achievable = min(peak_flops, (model_intensity*peak_bw)/1000.0)

                    ax.scatter(model_intensity, model_achievable, color="#00ffcc", s=150, zorder=5, label=f"Model Backbone Point")
                    ax.text(model_intensity*1.5, model_achievable, "Backbone", color="#00ffcc", fontweight="bold", fontsize=10)

                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel("Operational Intensity (FLOPs/Byte)", color="#d1d5db")
                    ax.set_ylabel("Performace Target (TFLOPS)", color="#d1d5db")
                    ax.tick_params(colors="#d1d5db")
                    ax.grid(True, which="both", linestyle="--", alpha=0.1)
                    ax.legend()
                    ax.set_title(f"Visualized Roofline Model Performance Limits", color="#ffffff", fontweight="bold")


                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error plotting Roofline models: {str(e)}")

    with bench_tabs[1]:
        st.markdown("### STREAM Memory Bandwidth Sweeps Dashboard")
        st.markdown(
            "Run simulated dynamic memory transfers (Copy, Scale, Add, Triad) representing STREAM benchmark configuration"
            "Compare calculated memory performance scaling profiles against selected device bus limits"
        )

        stream_col1, stream_col2 = st.columns([1,2])
        with stream_col1:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'>Sweep Constraints</div>"
                "</div>",
                unsafe_allow_html=True
            )

            target_sweep_gpu = st.selectbox("STREAM Mapped GPU Target", options=list(LabConfigs.GPU_PROFILES.keys()), key="stream_gpu_select")


            if st.button("EXECUTE STREAM BENCHMARK SWEEP"):
                st.toast("STREAM memory sweep calculation completed")

        with stream_col2:
            st.markdown(
                "<div class='glass-card'>"
                "<div class='glass-card-header'>STREAM Benchmark Performance</div>"
                "</div>",
                unsafe_allow_html=True
            )

            target_sweep_gpu = st.selectbox("STREAM Mapped GPU Target", options=list(LabConfigs.GPU_PROFILES.keys()), key="stream_gpu_select")

            if plt is not None:
                try:
                    fig, ax = plt.subplots(figsize=(6,4), facecolor="#0C0f16")
                    ax.set_facecolor("#1a1f2c")

                    ax.bar(stream_bw.keys(), stream_bw.values(), color="#3b8246", edgecolor="black", alpha=0.85)
                    ax.set_ylabel("Measured Transfer Bandwidth (GB/s)", color="#d1d5db")
                    ax.tick_params(colors="#d1d5db")
                    ax.grid(True, linestyle="--", alpha=0.1)
                    ax.set_title(f"STREAM Memory Bandwidth Sweep ({target_sweep_gpu})", color="#ffffff", fontweight="bold")

                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error drawing STREAM graphs: {str(e)}")

#DEEP LEARNING MODEL TRAINING (CONVERGENCE PHYSICS)
elif st.session_state.stage == 7:
    st.header("Stage 7 Deep Learning Staging & Convergence Training Cell")
    st.markdown(
        "Execute simulated stochastic gradient training epochs using physics-bases random convergence walks"
        "Review dynamic multi-task loss reductions, tracking metrics updates in real-time"
    )

    col1, col2= st.columns([1,2])

    with col1:
        st.markdown(
            "<div class='glass-card'>"
            "<div class='glass-card-header'>Training Control Panel </div>"
            "</div>",
            unsafe_allow_html=True
        )

        backbone = st.selectbox(
            "Model Backbone Weight size",
            options=list(LabConfigs.YOLO_BACKBONES.keys()),
            index=list(LabConfigs.YOLO_BACKBONES.keys()).index(st.session_state.model_backbone),
            disabled=st.session_state.training_running
        )
        st.session_state.model_backbone=backbone
        st.markdown(f"<span style='font-size: 0.8rem; color: #9ca3af;'>{LabConfigs.YOLO_BACKBONES[backbone]['description']}</span>", unsafe_allow_html=True)

        batch_scale = st.select_slider(
            "Batch Scale",
            options=[4,8,16,32,64],
            value=st.session_state.batch_size,
            disabled=st.session_state.training_running

        )

        st.session_state.batch_size = batch_scale

        lr_init = st.number_input(
            "Initial Learning Rate (lr0)",
            min_value=0.0001,
            max_value=0.5,
            value=st.session_state.learning_rate,
            step=0.001,
            format="%.5f",
            disabled=st.session_state.training_running
        )
        st.session_state.learning_rate = lr_init

        epochs = st.slider(
            "Epoch Limits",
            min_value=5,
            max_value=100,
            value=st.session_state.epochs_count,
            disabled=st.session_state.training_running
        )
        st.session_state.epochs_count = epochs

        opt = st.selectbox(
            "Weight Optimizer",
            options=["AdamW", "SGD", "RMSprop"],
            index=["AdamW", "SGD", "RMSprop"].index(st.session_state.optimizer_type),
            disabled=st.session_state.training_running
        )
        st.session_state.optimizer_type = opt

        decay = st.selectbox(
            "LR Scheduler type",
            options=["Cosine Annealing", "Step Decay", "Linear Warmup & Plateau"],
            index=["Cosine Annealing", "Step Decay", "Linear Warmup & Plateau"].index(st.session_state.lr_decay_type),
            disabled=st.session_state.training_running
        )
        st.session_state.lr_decay_type = decay

        st.divider()

        #Action trigger buttons
        if not st.session_state.training_running:
            #Require uploaded/selected dataset arget configuration
            dataset_ready = (st.session_state.uploaded_dataset_type is not None or st.session_state.active_dataset_preset is not None)

            if not dataset_ready:
                st.warning("Stage 2 Dataset requirements incomplete. Mount storage and parse configs")

            if st.button("EXECUTE MODEL TRAINING", disabled=not dataset_ready):
                st.session_state.training_running= True
                st.session_state.training_completed= False
                st.session_state.current_epoch = 0
                st.session_state.training_history = []
                st.session_state.epoch_log_feed = f"Starting active deep learning training loop... \nModel:{backbone}\nBatch scale: {batch_scale}\nOptimizer: {opt}\nStagin epochs counts: {epochs}\n\n"
                st.rerun()
        else:
            if st.button("HALT MODEL TRAINING"):
                st.session_state.training_running = False
                st.session_state.epoch_log_feed += "\n[WARNING] Training execution interrupted by user command\n"
                st.rerun()
    with col2:
        st.markdown(
            "<div class='glass-card'>"
            "<div class='glass-card-header'>Live Telemetry & Progress Trace</div>"
            "</div>",
            unsafe_allow_html=True
        )

        #Handle active training execution loop
        if st.session_state.training_running:
            curr_ep = st.session_state.current_epoch
            total_ep=st.session_state.epochs_count

            #Execute simulated epoch iteration steps
            if curr_ep < total_ep:
                curr_ep +=1
                st.session_state.current_epoch = curr_ep

                #Compare LR decays
                current_lr = DeepLearningPhysicsSimulator.calculate_lr_decay(
                    initial_lr=st.session_state.learning_rate,
                    epoch=curr_ep,
                    total_epochs=total_ep,
                    schedule_type=st.session_state.lr_decay_type
                )

                #Core simulation calculations
                step_metrics = DeepLearningPhysicsSimulator.simulate_epoch_step(
                    epoch=curr_ep,
                    total_epochs=total_ep,
                    backbone_scale=1.0 + (list(LabConfigs.YOLO_BACKBONES.keys()).index(st.session_state.model_backbone)*0.25),
                    batch_multiplier=st.session_state.batch_size / 16.0,
                    current_lr=current_lr,
                    seed=101
                )

                st.session_state.training_history.append(step_metrics)

                #Populate terminal logging reports
                log_line = f"Epoch {curr_ep:02d}/{total_ep:02d} | Box-Loss: {step_metrics['box_loss']:.4f} | Class-Loss: {step_metrics['class_loss']:.4f} | DFL-Loss: {step_metrics['dfl_loss']:.4f} | Total-Loss: {step_metrics['total_loss']:.4f} | mAP50: {step_metrics['mAP50']*100:.2f}% | LR: {step_metrics['lr']:.6f}\n"
                st.session_state.epoch_log_feed += log_line

                #Control Visual progress bars
                prog_percent = int((curr_ep/total_ep)*100)
                st.progress(curr_ep/total_ep, text=f"Converging Epochs: {curr_ep}/{total_ep} ({prog_percent}%)")

                #Render side-by-side metric badges
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                met_col1.metric("Box Regression Loss", f"{step_metrics['box_loss']:.4f}", delta=f"{step_metrics['box_loss']-st.session_state.training_history[-2]['box_loss']:.4f}" if len(st.session_state.training_history)>1 else None, delta_color="inverse")
                met_col2.metric("Classification Loss", f"{step_metrics['class_loss']:.4f}", delta=f"{step_metrics['class_loss']-st.session_state.training_history[-2]['class_loss']:.4f}" if len(st.session_state.training_history)>1 else None, delta_color="inverse")
                met_col3.metric("mAP50 Metric", f"{step_metrics['mAP50']*100:.2f}%", delta=f"{(step_metrics['mAP50']- st.session_state.training_history[-2]['mAP50'])*100:.2f}%" if len(st.session_state.training_history)>1 else None)
                met_col4.metric("mAP50-95 Metric", f"{step_metrics['mAP50_95']*100:.2f}%", delta=f"{(step_metrics['mAP50_95']- st.session_state.training_history[-2]['mAP50_95'])*100:.2f}%" if len(st.session_state.training_history)>1 else None)

                #Update Dynamic Graphs
                hist_df = pd.DataFrame(st.session_state.training_history)

                st.markdown("**Stochastic Loss Reduction Trajectory**")
                st.line_chart(hist_df[["mAP50","mAP50_95"]])

                #Interactive console logs frame
                st.markdown("**Real-Time Epochs Streaming Logs Console**")
                st.markdown(f"<div class='console-frame'>{st.session_state.epoch_log_feed.replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)

                #Trigger brief sleep to allow user visualization
                time.sleep(0.35)
                st.rerun()

            else:
                #Finalize loop executions
                st.session_state.training_running=False
                st.session_state.training_completed=True
                st.session_state.epoch_log_feed += "\n[SUCCESS] Deep Learning Convergence Completed. Checkpoints saved to '/runs/detect/weights/best.pt'\n"

                #Register mock weight files inside filesystem mappings
                hist_df = pd.DataFrame(st.session_state.training_history)
                best_acc = hist_df["mAP50_95"].max()
                csv_results = WeightsVaultExporter.generate_results_csv(st.session_state.training_history)

                fs.add_virtual_file("runs/detect/train_exp01/weights", "best.pt", "84.2 MB", "weights")
                fs.add_virtual_file("runs/detect/train_exp01/weights", "last.pt", "84.2 MB", "weights")
                fs.add_virtual_file("runs/detect/train_exp01", "results.csv", f"{len(csv_results)/1024:.2f} KB", "csv", csv_results)

                st.success("Neural training completed successfully. Serialized weight outputs added to Weights Vault")
                time.sleep(1.0)
                st.rerun()
        else:
            #Idle state
            if st.session_state.training_completed:
                st.success("Neural model training session is fully completed")

                hist_df = pd.DataFrame(st.session_state.training_history)

                #Render side-by-side metric badges
                last_metrics= st.session_state.training_history[-1]

                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                met_col1.metric("Final Box Loss", f"{last_metrics['box_loss']:.4f}")
                met_col2.metric("Final Class Loss", f"{last_metrics['class_loss']:.4f}")
                met_col3.metric("Final Peak mAP50", f"{hist_df['mAP50'].max()*100:.2f}%")
                met_col4.metric("Final Peak mAP50-95", f"{hist_df['mAP50_95'].max()*100:.2f}%")

                st.markdown("**Stochastic Loss Reduction Trajectory**")
                st.line_chart(hist_df[["box_loss", "class_loss", "total_loss"]])

                st.markdown("**Accuracy (mAP) Scaling Progress**")
                st.line_chart(hist_df[["mAP50", "mAP50_95"]])

                st.markdown("**Session Terminal Trace**")
                st.markdown(f"<div class='console-frame'>{st.session_state.epoch_log_feed.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            else:
                st.info("Stage 7 training dashboard ready. Configure parameters and click 'Execute Model Training'")
                st.code(
                    """
                    $ python train.py --weights yolov8s.pt --epochs 20 --batch 16

                    Workspace state: Idle. Waiting for execution trigger
                    """,
                    language="bash"
                )
#EDGE COMPILERS & FASTAPI CODES
elif st.session_state.stage ==8:
    st.header("Stage 8: Inference Compiler & FastAPI Docker Vault")
    st.markdown(
        "Compile production-ready inference endpoints and system containers configurations"
        "Generate optimized FastAPI scripts and Dockerfiles supporting target accelerators parameters"
    )

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown(
            "<div class='glass-card'>"
            "<div class='glass-card-header'>Code Compiler Parameters</div>"
            "</div>",
            unsafe_allow_html=True
        )

        target_weights = st.text_input("Active Target Weights Name", value=st.session_state.model_backbone)
        active_classes = st.session_state.classes_detected if st.session_state.classes_detected else ["class1", "class2"]

        st.markdown("#### Mapped Class Classes tags:\n" + "\n".join([f"-`{c}`"for c in active_classes]))

        #Interactive PyTorch script compiler
        st.divider()
        st.markdown("#### Native PyTorch Prediction Script Code (`inference_run.py`)")
        torch_code = DeploymentScriptGenerator.generate_pytorch_inference(target_weights, active_classes)
        st.code(torch_code, language="python")

        fastapi_code = DeploymentScriptGenerator.generate_fast_api_code(target_weights, active_classes)
        if st.button("SAVE fastapi_server.py to WORKSPACE"):
            fs.add_virtual_file("configs", "fastapi_server.py", f"{len(fastapi_code)/1024:.2f} KB", "python", fastapi_code)
            st.success("fastapi_server.py successfully saved to drive storage")

            #Mapped C++ predictions generator
            st.divider()
            st.markdown("#### High-Performance C++ OpenCV DNN Prediction Engine")
            cpp_code = DeploymentScriptGenerator.generate_cpp_inference(target_weights, active_classes)
            st.code(cpp_code, language="cpp")

            st.markdown("---")
            st.markdown("### Dockerfile Staging Container Configuration")
            st.markdown(
                "Generate production-ready multi-stage Docker configurations optimized for target GPU and"
                "PCIe systems mapped in Stage 1"
            )

            docker_code = DeploymentScriptGenerator.generate_dockerfile(target_weights)
            st.code(docker_code, language="dockerfile")

            if st.button("SAVE Dockerfile TO WORKSPACE", key="save_dockerfile"):
                fs.add_virtual_file("configs", "Dockerfile", f"{len(docker_code)} B", "dockerfile", docker_code)
                st.success("Dockerfile successfully saved to drive storage")

#ARTIFACT VAULT & DOWNLOAD GATEAWAY
elif st.session_state.stage ==9:
    st.header("Stage 9: Artifact Vault & Model Exporter Gateway")
    st.markdown(
        "Access your compiled model checkpoints directly out of the application sandbox"
        "Convert PyTorch weights. `.pt` binaries into optimized ONNX execution formats, select TensorRT"
        "INT8 engine calibration matrices, and export container packages"
    )

    hist_df = pd.DataFrame(st.session_state.training_history)
    best_accuracy = hist_df["mAP50_95"].max() if not hist_df.empty else 0.82

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown(
            "<div class='glass-card'>"
            "<div class='glass-card-header'> Serialized Checkpoints Vault</div>"
            "</div>",
            unsafe_allow_html=True
        )

        checkpoints = [
            {"File Name": "best.pt", "Size":"84.2 MB", "Accuracy Target (mAP50)":f"{hist_df['mAP50'].max()*100:.2f}%" if not hist_df.empty else "89.2%", "Description": "Peak validation accuracy checkpoint binary"},
            {"File Name": "last.pt", "Size": "84.2 MB", "Accuracy Target (mAP50)":f"{hist_df['mAP50'].iloc[-1]*100:.2f}%" if not hist_df.empty else "87.4%", "Description": "Final epoch training state weights" },
            {"File Name": "results.csv", "Size": "4.2 KB", "Accuracy Target (mAP50)":"-","Description": "Epoch-by-Epoch loss metric log sheet"}
        ]
        st.table(checkpoints)

        st.divider()

        st.markdown("Dynamic Zip Archive Assembly")
        st.markdown("Package config YAMLs, loss metric sheets, and best accuracy checkpoints into a single downloadable vault")

        results_csv_data = WeightsVaultExporter.generate_results_csv(st.session_state.training_history)

        try:
            zip_bytes = WeightsVaultExporter.package_vault_zip(
                yaml_config=st.session_state.dataset_yaml_data if st.session_state.uploaded_dataset_type == "yaml" else "names: [classA, classB]",
                results_csv=results_csv_data,
                best_accuracy=best_accuracy,
                backbone_name=st.session_state.model_backbone
            )

            st.download_button(
                label="DOWNLOAD EXPORT WORKSPACE VAULT (.ZIP)",
                data=zip_bytes,
                file_name=f"DL_Workspace_Export_{st.session_state.model_backbone.replace('.pt','')}.zip",
                mime="application/zip"
            )
        except Exception as e:
            st.error(f"Error compiling download zip container: {str(e)}")

        st.divider()

        st.markdown("### Generated Model Card Cardboard")
        st.markdown(
            f"""
            - **Model Architecture Backbone**: `{st.session_state.model_backbone}`
            - **Epochs run**: `{st.session_state.epochs_count}` epochs
            - **Final Validation Loss**: `{"{:.4f}".format(hist_df['total_loss'].iloc[-1]) if not hist_df.empty else "0.3412"}`
            - **Quantization Compatibility Check**: FP16 / INT8 Calibaration Cache Available
            - **Execution Signature Hash**: `{uuid.uuid4().hex[:12].upper()}`
            """
        )

    with col2:
        st.markdown(
            "<div class='glass-card'>"
            "<div class='glass-card-header'>Inference Engines Optimizer Compiler </div>"
            "</div>",
            unsafe_allow_html=True
        )

        target_format = st.selectbox(
            "Export Compiler Target",
            options=["ONNX (FP16)", "TensorRT (INT8 Quantized)", "CoreML (FP16 macOS Target)"]
        )

        st.markdown("#### Engine Optimization parameters")
        if target_format == "ONNX (FP16)":
            st.checkbox("Enable FP16 half precision quantization layers", value=True)
            st.checkbox("Enable constant-folding optimization pass", value=True)
            st.slider("Opset Level Target", min_value=9, max_value=16, value=12)
        elif target_format == "TensorRT (INT8 Quantized)":
            st.checkbox("Run fll INT8 quantization calibration cache run", value=True)
            st.selectbox("Calibration Dataset Map", options=["COCO Validation Sample (500 Images)", "Active Staged train splits dataset"])
            st.slider("CUDA Compute Capability Target", min_value=6.0, max_value=9.0, value=8.6, step=0.1)
        elif target_format == "CoreML (FP16 macOS Target)":
            st.checkbox("Include flexible dynamic image input tensor shapes", value=False)
            st.checkbox("Apply NMS bounding box layers directly to Graph", value=True)

        if st.button("COMPILE TARGET EDGE INFERENCE MODEL"):
            st.session_state.onnx_export_status = "COMPILING..."
            st.session_state.export_logs=f"Initializing system compiler backend: {target_format}...\n"

            with st.status(f"Compiling model graph to {target_format} target...", expanded=True) as status:
                st.write("Loading PyTorch weight graphs matrices...")
                time.sleep(0.8)
                st.write("Fusing convolution and normalization layers scales...")
                st.session_state.export_logs += "[INFO] Fused Conv2D + BatchNorm2D node structures \n"
                time.sleep(0.7)
                st.write("Performing constant foldin calculations...")
                st.session_state.export_logs += "[INFO] Optimization passes completed: Graph pruned from 312 nodes to 242 nodes \n"
                time.sleep(0.9)

                if "INT8" in target_format:
                    st.write("Running calibration cache quantizations maps...")
                    st.session_state.export_logs += "[QUANTIZATION] Calibrating dynamic weights tensor scaling distributions...\n"
                    time.sleep(1.2)
                    st.session_state.export_logs += "[QUANTIATION] Average KL Divergence error profile: 0.00241 \n"

                st.write("Writing engine artifacts binaries to run directory...")
                time.sleep(0.5)
                status.update(label="Edge Model Export Succeeded. Target formats are active", state="complete")
            
            st.session_state.onnx_export_status = "EXPORTED"
            st.session_state.active_compilation_details = WeightsVaultExporter.get_compilation_details(target_format, st.session_state.model_backbone)
            st.rerun()

        if st.session_state.onnx_export_status == "EXPORTED":
            st.success(f"Target Graph Optimization to {target_format} Succeeded")

            st.markdown("#### Mapped Compilation Details Profile")
            comp_df = pd.DataFrame(
                list(st.session_state.active_compilation_details.items()),
                columns=["Optimization Parameter", "Active Value"]
            )
            st.table(comp_df)

            #Mock terminal logs output
            st.markdown("#### Compilter Execution Terminal Logs")
            log_text = st.session_state.export_logs + f"\n[SUCCESS] Engine weights exported. File Saved: /runs/detect/train_exp01/{st.session_state.active_compilation_details.get('Export File Name', 'engine.file')}\n"
            st.markdown(f"<div class='console-frame'>{log_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

            #ADD EXPORTED FILE TO SYSTEM MAPPING
            fs.add_virtual_file(
                "runs/detect/train_exp01",
                st.session_state.active_compilation_details.get('Export File Name','engine.file'),
                "12.4 MB",
                "compiled_model"
            )
        else:
            st.info("Optimize  inference architectures above to compile high-speed edge hardware targets")

#EXTREMELY DETAILED SYSTEM LOGS AREA (STATE CONTROL PANEL)

st.markdown("---")
st.markdown("###Master Systems Telemetry Logs Engine")
st.markdown(
    "Trace host container operations logs. Search and filter logs files parameters by"
    "entering search strings"
)

log_filter = st.text_input("Filter logs containing string", value="", placeholder="Type filter e.g., INFO, SUCCESS, CUDA...")

filtered_logs = [log for log in st.session_state.system_logs_list if log_filter.lower() in log.lower()]

log_frame_text = "<br>".join(filtered_logs)
st.markdown(f"<div class='console-frame'>{log_frame_text}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    pass




        










   




