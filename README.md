# AI Medical Department — Multi-Agent System

A multi-agent clinical decision support system that simulates a hospital consultation workflow. Each agent specializes in a medical domain (Radiology, Cardiology, Oncology), with findings synthesized and surfaced to physicians via an interactive interface.

**Dataset:** [OpenM3Chest](https://openm3chest.org) (NLST-based CT + clinical data)  
**Backbone:** [MedGemma 1.5-4B-IT](https://huggingface.co/google/medgemma-1.5-4b-it) — one base model, three LoRA adapters  
**Infra:** VastAI GPU containers (training + serving)

---

## System Architecture

```
CT Scan (.npy) + Clinical Data (JSON)
          │
          ▼
   [ Data Prep Agent ]
   CT slicing · Lungmask · clinical_to_text()
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
[Radio] [Cardio] [Onco]   ← MedGemma + LoRA (domain-specific)
    └─────┼─────┘
          │ hidden states          │ text answers
          ▼                        ▼
  [ Synthesis Agent ]       [ Report Agent ]
    SVFM cross-modal         structured clinical report
    fusion → embedding
          │
    ┌─────┘
    ▼
  FAISS retrieval             Doctor Interface (RAG + MedASR)
  similar cases               physician Q&A via text or voice
```

---

## Repository Structure

```
ai-medical-dept/
├── agents/              # Core agent logic
│   ├── data_prep/       # CT preprocessing, segmentation, clinical text
│   ├── radiology/       # lora_radiology — 7 task files
│   ├── cardiology/      # lora_cardiology — CVD diagnosis + mortality
│   ├── oncology/        # lora_oncology — 6-year cancer risk
│   ├── synthesis/       # SVFM cross-modal fusion module
│   ├── report/          # Structured report generator
│   └── doctor_interface/# RAG + MedASR voice input
├── training/            # QLoRA fine-tuning scripts per agent
├── serving/             # FastAPI inference server + LoRA manager
├── retrieval/           # FAISS index builder and query
├── configs/             # YAML configs for training & serving
├── docker/              # Dockerfiles for VastAI (train / serve)
├── scripts/             # VastAI setup, training launch, index build
├── tests/               # Unit and integration tests
└── notebooks/           # EDA and baseline evaluation
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/<your-org>/ai-medical-dept.git
cd ai-medical-dept
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: HF_TOKEN, DATA_ROOT, CHECKPOINT_DIR
```

### 3. Fine-tune an agent (VastAI)

```bash
# Upload to VastAI instance and run:
bash scripts/vastai_train.sh radiology
```

### 4. Start inference server

```bash
bash scripts/serve.sh
# API available at http://localhost:8000
```

---

## Agents

| Agent | LoRA | Training Data | Samples |
|---|---|---|---|
| Radiology | `lora_radiology` | 7 task files (chest_abn, covid19, nodule_*) | ~320k |
| Cardiology | `lora_cardiology` | CVD_diagnosis + CVD_mortality | ~47k |
| Oncology | `lora_oncology` | lung_cancer_risk | ~46k |
| Synthesis | from scratch | Multi-task labels from all files | all |
| Report | none | — | — |
| Doctor Interface | none (base MedGemma) | — | — |

---

## Team

| Name | Role |
|---|---|
| **Ung Hoàng Long** | Team Lead |
| Đinh Trần Duy Trường | Member |
| Trần Bảo Trân | Member |
| Nguyễn Lê Thanh Minh | Member |

---

## License

Research use only. MedGemma is subject to [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms).
