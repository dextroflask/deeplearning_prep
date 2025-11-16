# Deep Learning Mastery — 4 Week Intensive Program

This repository contains a **rigorous 4-week learning plan** to rebuild and advance mastery in:

- Machine Learning  
- Deep Learning  
- Advanced Optimization  
- Self-Supervised Learning (SSL)  
- Contrastive Learning (SimCLR, MoCo, BYOL, SimSiam, DINO, iBOT)  
- Few-Shot Learning (ProtoNet, MatchingNets, RelationNets, CLIP FSL)  
- Transformers & Vision Transformers (ViT, DeiT, Swin)  
- Explainable AI (Grad-CAM, IG, SHAP, LIME, ViT Rollout)  
- Advanced Computer Vision Architectures  
- Diffusion Models  
- ML System Design  

All weeks include **specific topics + deliverables**.  
A complete, curated **resource list** is provided at the bottom.

---

# 📅 4-Week Schedule Overview

---

# Week 1 — Mathematics for Machine Learning  
*Goal: rebuild high-level mathematical foundations required for DL, SSL, contrastive methods, diffusion, transformers, and advanced optimization.*

---

## Topics

### Linear Algebra
- Matrix calculus, Jacobians, Hessians  
- Eigendecomposition, SVD  
- Frobenius, L1/L2 norms  
- Quadratic forms  
- Vector-Jacobian products (VJP), JVP  

### Probability & Statistics
- PDFs, PMFs, CDFs  
- Expectation, variance, covariance  
- MLE, MAP  
- Bayesian inference basics  
- KL divergence (derive it)  
- Cross-entropy = NLL connection  

### Optimization Theory
- Convexity, Lipschitz smoothness  
- Gradient descent convergence  
- Lagrange multipliers  
- Stochastic gradient behavior  

### Information Theory
- Entropy  
- Mutual Information I(X;Y)  
- InfoNCE theoretical foundation  

---

## Deliverables
- `matrix_calculus.ipynb` (derivatives of softmax, CE, dot products, quadratic forms)  
- `probability_kl.ipynb` (KL for Gaussians, CE, expectations)  
- `optimization_algorithms.ipynb` (GD, SGD, Momentum, RMSProp, Adam)  
- **1-page Mathematics Summary PDF**

---

# Week 2 — Core Deep Learning + Loss Functions + Training Tricks

---

## Topics

### Neural Networks
- Backprop through computational graphs  
- Activations (ReLU, LeakyReLU, GELU, Swish)  
- Initializations: Xavier, Kaiming  
- Normalizations: BatchNorm, LayerNorm, GroupNorm, RMSNorm  

### Loss Functions
- MSE, MAE  
- Cross-entropy  
- KL Divergence Loss  
- Focal Loss  
- Label Smoothing  
- Triplet Loss  
- Contrastive Loss  
- InfoNCE  
- NT-Xent (SimCLR)  
- Dice Loss  
- IoU Loss  

### Modern Training Techniques
- Learning rate schedules (Warmup, Cosine, One-Cycle)  
- AdamW, SGD-Momentum, RMSProp  
- Mixed Precision Training (AMP)  
- Gradient Clipping  
- Sharpness-Aware Minimization (SAM)  

### CNN Architectures
- ResNet  
- EfficientNet  
- ConvNeXt  
- Depthwise Separable & Dilated Convolutions  

---

## Deliverables
- `resnet_from_scratch` implementation + CIFAR-10 (85%+)  
- `loss_functions.ipynb` (all loss functions implemented & compared)  
- `modern_training/optimizer_benchmarks.ipynb`  
- **1-page Training Tricks Summary**

---

# Week 3 — SSL, Contrastive Learning, DINO, FSL & Transformers

---

## Topics

### Contrastive Learning
- Positive/Negative pairs  
- NT-Xent  
- Momentum encoders (MoCo)  
- Queue-based negatives  
- Temperature scaling  

### Self-Supervised Learning
Study in exact order:
1. SimCLR  
2. MoCo v2  
3. BYOL  
4. SimSiam  
5. DINO  
6. iBOT  

### Few-Shot Learning (FSL)
- Metric learning  
- Prototypical Networks  
- Matching Networks  
- Relation Networks  
- CLIP as FSL model  

### Transformers
- Scaled dot-product attention (derive fully)  
- Multi-Head Attention  
- Positional Encoding  
- Encoder vs Decoder  
- ViT  
- DeiT  
- Swin Transformer  

### Loss Functions for SSL/FSL
- InfoNCE  
- Symmetrized losses  
- Multi-crop loss (DINO)  
- Masked image modeling losses  

---

## Deliverables
- `simclr/` full pipeline + NT-Xent implementation  
- CIFAR-10 linear probe with **75%+ accuracy**  
- `dino/` student-teacher implementation (centering, temperature)  
- `few_shot/protonet.ipynb`  
- `transformers/attention_from_scratch.ipynb`  
- **2-page SSL Comparison Document**

---

# Week 4 — Explainable AI, Advanced Vision, Diffusion, ML System Design

---

## Topics

### Explainable AI (XAI)
- Grad-CAM  
- Grad-CAM++  
- Integrated Gradients  
- SHAP  
- LIME  
- Attention rollout for ViTs  

### Advanced Vision Architectures
- U-Net  
- DeepLab v3  
- Mask R-CNN  
- DETR  
- YOLOv8  

### Diffusion Models
- Forward diffusion process  
- Variance schedules  
- Reverse denoising process  
- Classifier-free guidance  
- U-Net for diffusion  

### ML System Design
- Training pipeline design  
- Feature stores  
- Drift detection  
- Monitoring (prediction drift, data drift)  
- Deployment strategies  
- Vector databases for embeddings (FAISS, Milvus)  

---

## Deliverables
- `grad_cam.py` + `integrated_gradients.py`  
- XAI demos on ResNet + ViT  
- One advanced CV model implemented (U-Net / DETR)  
- `diffusion/ddpm_scratch.ipynb`  
- `ml_system_design_doc.md`  
- **CLIP-like Retrieval Demo** (image–text similarity)

---

# 📚 Resource List (Curated)

---

## Mathematics
- *The Matrix Cookbook*  
- *All of Statistics* — Wasserman  
- Boyd & Vandenberghe — *Convex Optimization*  
- MIT 6.S191 + 3Blue1Brown  
- CS229 Math Review  

---

## Deep Learning
- *Deep Learning Book* — Goodfellow  
- CS231n (CNNs, Backprop)  
- FastAI — Modern DL  

---

## Loss Functions
- Focal Loss (paper)  
- Dice Loss (paper)  
- InfoNCE — CPC Paper  
- Lil’Log — Contrastive Learning Blog  

---

## Self-Supervised Learning
- SimCLR  
- MoCo v2  
- BYOL  
- SimSiam  
- DINO  
- iBOT  
- Lil’Log SSL Series  

---

## Few-Shot Learning
- Prototypical Networks  
- Matching Networks  
- Relation Networks  
- CLIP (OpenAI)  

---

## Transformers
- Attention Is All You Need  
- ViT  
- DeiT  
- Swin Transformer  
- Annotated Transformer  

---

## Explainable AI & Advanced CV
- Grad-CAM  
- Grad-CAM++  
- Integrated Gradients  
- DeepLab v3  
- DETR  

---

## Diffusion Models
- DDPM  
- HuggingFace Diffusers  
- Lil’Log Diffusion Series  

---

## ML System Design
- *Designing Machine Learning Systems* — Chip Huyen  
- Google Rules of ML  
- W&B Engineering Blogs  
- FAISS Documentation  

---

# ⭐ End Goal

By the end of this 4-week plan, you will have:

- A complete mathematics base  
- Deep mastery of CNNs, Transformers, SSL, FSL  
- Implemented **SimCLR, DINO, ProtoNet, ViT, Grad-CAM, DDPM**  
- Strong ML system design knowledge  
- A polished, multi-folder GitHub portfolio  

---
