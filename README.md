# Deep Learning Mastery — 4 Week Intensive Program

This repository contains a **rigorous and structured 4-week plan** designed to rebuild, deepen, and master:

- Machine Learning & Deep Learning  
- Mathematics for ML  
- Advanced Optimization  
- Self-Supervised Learning (SimCLR, MoCo, BYOL, SimSiam, DINO, iBOT)  
- Few-Shot Learning (ProtoNet, MatchingNets, RelationNets, CLIP)  
- Transformers & Vision Transformers (ViT, DeiT, Swin)  
- Explainable AI (Grad-CAM, IG, SHAP, LIME)  
- Diffusion Models  
- ML System Design  

Each week includes:  
✔ **Exact topics**  
✔ **Precise deliverables**  
✔ **Actionable, chapter-specific and section-specific resources**  
✔ **Paper-specific instructions**  

---

# 📅 4-Week Schedule Overview

---

# **WEEK 1 — Mathematics for Machine Learning**  
*Goal: Rebuild mathematical foundations required for DL, SSL, transformers & advanced optimization.*

---

## **TOPICS**

### **1. Linear Algebra**
- Matrix calculus (Jacobian, Hessian)
- Eigendecomposition, SVD
- Vector norms & induced matrix norms
- Quadratic forms  
- Vector-Jacobian product (VJP), JVP  

**Resources (Specific Chapters):**
- **Matrix Cookbook** → Sections: *Derivatives*, *Identities*  
- **Deep Learning Book (Goodfellow)** → Chapter **2**  
- **CS229 Linear Algebra Review** → Entire PDF  

**Action:** Work through **all derivative identities** & implement them in Week-1 notebooks.

---

### **2. Probability & Statistics**
- PDFs, PMFs, CDFs  
- Expectation, variance, covariance matrix  
- MLE, MAP  
- KL Divergence derivation  
- CE–NLL connection  

**Resources:**
- **All of Statistics (Wasserman)** → Chapters **1, 2, 3, 7, 8, 11**  
- **CS229 Probability Review**  

**Action:** Solve 10–15 exercises from each chapter (summary only in notebook).

---

### **3. Optimization Theory**
- Convexity & smoothness  
- Gradient descent convergence  
- Lagrange multipliers  
- Stochastic gradients (variance analysis)  

**Resources:**
- **Convex Optimization (Boyd)** → Chapters **1–4**, **9.1–9.3**  
- MIT 6.036 Optimization Lectures  

---

### **4. Information Theory**
- Entropy  
- Cross entropy  
- Mutual Information  
- InfoNCE derivation  

**Resources:**
- *Elements of Information Theory (Cover & Thomas)* → Chapters **1–2**  
- CPC paper (Section: *Contrastive Predictive Coding Loss*)  

---

## **DELIVERABLES (VERY CLEAR)**

### **1. `matrix_calculus.ipynb`**
Implement by hand:
- Gradient of softmax  
- Gradient of CE loss  
- Gradient of quadratic form xᵀAx  
- Derivative of sigmoid, tanh, GELU  

---

### **2. `probability_kl.ipynb`**
Include:
- KL divergence between two Gaussians (derive)  
- CE = NLL proof (write steps)  
- MLE of mean & variance for Normal distribution  
- Bayes rule toy examples  

---

### **3. `optimization_algorithms.ipynb`**
Implement from scratch:
- GD, SGD, Momentum  
- RMSProp  
- Adam  
- Plot convergence curves on a quadratic bowl & a simple NN  

---

### **4. `math_summary.pdf`**  
One-page notes (your own summary).

---

# **WEEK 2 — Core Deep Learning + Loss Functions + Modern Training**

---

## **TOPICS**

### **1. Neural Network Foundations**
- Backprop through computation graph  
- Xavier & Kaiming initialization  
- Normalization techniques: BN, LN, GN, RMSNorm  

**Resources:**
- **Deep Learning Book** → Chapters **6, 7, 8**  
- **CS231n** → Backprop Notes  

---

### **2. Loss Functions (Deep Dive)**
You must **derive the formula**, **implement it**, and **train a model** using each.

Losses to cover:
- MSE, MAE  
- Cross-entropy  
- KL-divergence  
- Label smoothing  
- Focal Loss → *RetinaNet Paper Section 3*  
- Triplet Loss → *FaceNet Paper Section 4.2*  
- Contrastive Loss → *SimCLR Paper Section 3.4*  
- InfoNCE → *CPC Paper Section 2.2*  
- NT-Xent → *SimCLR Paper Equation (1)*  
- Dice Loss  
- IoU Loss  

---

### **3. Modern Training Techniques**
- SGD-M, AdamW  
- Cosine LR, Warmup  
- AMP (Mixed Precision)  
- Gradient Clipping  
- SAM (Section 3 of the SAM Paper)  

---

### **4. CNN Architectures**
Implement & compare:
- ResNet (ResNet paper Sections 3.3–4)  
- EfficientNet (compound scaling idea only)  
- ConvNeXt (Sections 3–4)  

---

## **DELIVERABLES**

### **1. `resnet_from_scratch/`**
- Build ResNet-18 from scratch (PyTorch)  
- Train on CIFAR-10  
- **Required: 85%+ test accuracy**  
- Include training curves  

---

### **2. `loss_functions/`**
Notebooks:
- `basic_losses.ipynb` → MSE, CE, KL  
- `ranking_losses.ipynb` → Triplet, Contrastive  
- `segmentation_losses.ipynb` → Dice, IoU  
- `ssl_losses.ipynb` → InfoNCE, NT-Xent  

Each notebook must include:
- Formula  
- Gradient sketch  
- PyTorch implementation  
- Comparison plots  

---

### **3. `modern_training/optimizer_benchmarks.ipynb`**
- Benchmark Adam, AdamW, SGD-M, RMSProp  
- Compare convergence + generalization  

---

### **4. `training_tricks_summary.pdf`**
Your own summary.

---

# **WEEK 3 — SSL, Contrastive Learning, DINO, FSL & Transformers**

---

## **TOPICS**

### **1. Contrastive Learning**
Study **AND reproduce**:
- SimCLR → Sections **3.1–3.4**  
- MoCo v2 → Sections **3–4**  
- Temperature scaling analysis  
- Momentum encoder update (MoCo Eq. (2))  

---

### **2. Self-Supervised Learning (SSL) Pipelines**
**Read + extract key mechanisms:**

| Method | Required Sections to Study |
|--------|---------------------------|
| **SimCLR** | Entire Section 3 (architecture, loss, augmentations) |
| **MoCo v2** | Sections 2–4 |
| **BYOL** | Sections 1–4 (stop-grad mechanism) |
| **SimSiam** | Sections 3–5 (no negatives) |
| **DINO** | Sections 3–4 (student-teacher, centering, temperature) |
| **iBOT** | Sections 2–4 (MIM + contrastive) |

---

### **3. Few-Shot Learning**
Study and implement:
- Prototypical Networks → Sections 2–3  
- Matching Networks → Sections 2–4  
- Relation Networks → Sec 3  
- CLIP FSL → Zero-shot retrieval (OpenAI CLIP paper Sections 2–4)  

---

### **4. Transformers**
Study and implement:
- Attention derivation → *Attention Is All You Need* Section **3.2**  
- MHA → Section **3.2.2**  
- Positional Encodings → Section **3.5**  
- ViT → Sections **3–4**  
- DeiT → Sections **3–5**  
- Swin → Sections **3–4**  

---

## **DELIVERABLES**

### **SIMCLR**
Folder: `simclr/`
- `nt_xent.py` — your implementation  
- `train_simclr.py` — full training  
- `augmentations.py` — SimCLR augmentations  
- Train on CIFAR-10  
- **Linear probe accuracy ≥ 75%**  

---

### **DINO**
Folder: `dino/`
- Implement student–teacher  
- Implement centering  
- Implement temperature sharpening  
- Train on CIFAR-10 for 100 epochs  
- Extract features + run KMeans (k=10)  

---

### **Few-Shot**
- `protonet.ipynb`  
- `matching_nets.ipynb`  
- `relation_nets.ipynb`  
- Evaluate 5-way 1-shot accuracy  

---

### **Transformers**
- `attention_from_scratch.ipynb` → derive attention step by step  
- `tiny_transformer.py` → build a mini Transformer  
- `vit_scratch.py` → patchify + encoder  

---

### **SSL Comparison**
- `ssl_comparison.pdf` → 2 pages summarizing all methods  

---

# **WEEK 4 — XAI, Advanced CV, Diffusion, ML System Design**

---

## **TOPICS**

### **Explainable AI**
Study:  
- Grad-CAM  
- Grad-CAM++  
- Integrated Gradients (IG paper Section 3)  
- SHAP  
- LIME  
- Attention rollout for ViTs  

---

### **Advanced CV Architectures**
Study specific sections:
- U-Net → Sections **2–3**  
- DeepLab v3 → Section **3**  
- Mask R-CNN → Section **3**  
- DETR → Sections **4–5**  
- YOLOv8 → architecture overview  

---

### **Diffusion Models**
Study & reproduce:  
- DDPM → Sections **2–3**  
- Reparameterization of noise schedule  
- Reverse denoising process  
- Classifier-free guidance  

---

### **ML System Design**
Study and create:
- Data pipelines  
- Feature stores  
- Model monitoring (prediction drift, data drift)  
- Deployment patterns  
- Vector DBs for embeddings  

---

## **DELIVERABLES**

### **1. Explainable AI Implementation**
Folder: `explainable_ai/`
- `grad_cam.py` + heatmaps  
- `integrated_gradients.py`  
- SHAP demo on ResNet  
- ViT attention rollout  

---

### **2. Advanced Vision Model**
Implement one:
- U-Net  
- DeepLab v3  
- DETR  

Include:
- Training script  
- Evaluation metrics  

---

### **3. Diffusion**
- `ddpm_scratch.ipynb`  
Includes:  
- Forward diffusion  
- Reverse sampling  
- Generate sample images  

---

### **4. ML System Design**
- `ml_system_design_doc.md`  
- Contains: pipelines, feature store, deployment graph  

---

### **5. CLIP Retrieval Demo**
Folder: `clip_retrieval_demo/`
- `encode_image.py`  
- `encode_text.py`  
- `retrieval_demo.ipynb`  

---

# 📚 RESOURCE LIST (With Chapter/Section Specific Guidance)

---

## **Mathematics**
- *Matrix Cookbook* → Derivatives + identities  
- *All of Statistics* → Ch 1–3, 7–8, 11  
- *Convex Optimization (Boyd)* → Ch 1–4, 9.1–9.3  
- CS229 Math Review → Entire  
- *Elements of Information Theory* → Ch 1–2  

---

## **Deep Learning**
- *Deep Learning Book* → Ch 6–8  
- CS231n → Backprop + CNNs  
- FastAI → Modern training methods  

---

## **Loss Functions-Related Papers**
- **Focal Loss (RetinaNet)** → Section 3  
- **Triplet Loss (FaceNet)** → Section 4.2  
- **Dice Loss** → 2016 medical imaging paper (Section 2)  
- **NT-Xent** → SimCLR Eq. (1)  
- **InfoNCE** → CPC Section 2.2  

---

## **Self-Supervised Learning Papers**
- **SimCLR** → Sections 3–4  
- **MoCo v2** → Sections 2–4  
- **BYOL** → Sections 1–4  
- **SimSiam** → Sections 3–5  
- **DINO** → Sections 3–4  
- **iBOT** → Sections 2–4  

---

## **Few-Shot Learning Papers**
- **Prototypical Networks** → Sections 2–3  
- **Matching Networks** → Sections 2–4  
- **Relation Networks** → Section 3  
- **CLIP** → Sections 2–4  

---

## **Transformers Papers**
- **Attention Is All You Need** → Sections 3.2, 3.5  
- **ViT** → Sections 3–4  
- **DeiT** → Sections 3–5  
- **Swin** → Sections 3–4  
- **Annotated Transformer** → Entire walkthrough  

---

## **Explainable AI**
- **Grad-CAM** → Entire paper  
- **Grad-CAM++** → Sec 3  
- **Integrated Gradients** → Sec 3  
- **SHAP Docs**  
- **LIME Docs**  

---

## **Diffusion**
- **DDPM** → Sections 2–3  
- HF Diffusers → Tutorials  

---

## **ML System Design**
- **Chip Huyen** → Ch 2–7  
- **Google Rules of ML** → Entire  
- **W&B Articles**  
- **FAISS Docs**  

---

# ⭐ END GOAL

By the end of this program you will have:

- Rebuilt complete ML mathematics  
- Implemented **ResNet, SimCLR, DINO, ProtoNet, Transformers, DDPM**  
- Learned XAI and ML System Design  
- Completed a polished GitHub portfolio suitable for ML Engineer roles  

---
