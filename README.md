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

Repo này chứa toàn bộ phần **AI/ML**: data pipeline, training, serving, và agent logic.
Phần **backend API và frontend** sẽ được phát triển trong một repo riêng biệt.

```
ai-medical-dept/
├── agents/                  # Logic từng agent
│   ├── data_prep/           # Tiền xử lý CT (.npy slicing) + clinical text
│   ├── radiology/           # LoRA radiology — nhận diện bất thường phổi
│   ├── cardiology/          # LoRA cardiology — chẩn đoán CVD + mortality
│   ├── oncology/            # LoRA oncology — dự đoán nguy cơ ung thư 6 năm
│   ├── synthesis/           # SVFM cross-modal fusion (kết hợp output các agent)
│   ├── report/              # Tạo báo cáo lâm sàng có cấu trúc
│   └── doctor_interface/    # RAG + MedASR (giao tiếp bác sĩ qua văn bản/giọng nói)
├── training/                # Script QLoRA fine-tune từng agent trên VastAI
├── serving/                 # FastAPI inference server + quản lý LoRA adapter
│                            # (giữ lại cấu trúc, chưa active — backend sẽ ở repo riêng)
├── retrieval/               # Xây dựng và truy vấn FAISS index (RAG)
├── configs/                 # YAML config cho training và serving
├── docker/                  # Dockerfile cho VastAI (train / serve)
├── scripts/                 # Script setup VastAI, chạy training, build index
├── tests/                   # Unit test và integration test
└── notebooks/               # EDA và đánh giá baseline
```

> **Phân công repo:**
> - `ai-medical-dept` (repo này) — AI pipeline: data, training, agents, serving
> - `ai-medical-web` (repo riêng, tạo sau) — Backend API + Frontend website

---

## Quickstart

### 1. Clone & cài môi trường

```bash
git clone https://github.com/UngHoangLong/ai-medical-dept.git
cd ai-medical-dept
python -m venv venv && source venv/bin/activate
pip install -r requirements/base.txt
```

### 2. Cấu hình environment

```bash
cp .env.example .env
# Điền vào: HF_TOKEN (HuggingFace), DATA_ROOT (đường dẫn data local)
```

### 3. Load labels từ HuggingFace

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="UngLong/openm3chest-labels",
    repo_type="dataset",
    local_dir="data/"
)
```

### 4. Fine-tune một agent (chạy trên VastAI)

```bash
# Sau khi thuê GPU trên VastAI và clone repo vào instance:
bash scripts/vastai_train.sh radiology
```

> Xem hướng dẫn làm việc nhóm và quy trình Git tại [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Agents

| Agent | LoRA | Dữ liệu training | Số samples |
|---|---|---|---|
| Radiology | `lora_radiology` | chest_abn (×54), nodule_* | ~320k |
| Cardiology | `lora_cardiology` | CVD_diagnosis + CVD_mortality | ~47k |
| Oncology | `lora_oncology` | lung_cancer_risk | ~46k |
| Synthesis | from scratch | Multi-task labels từ tất cả file | tất cả |
| Report | — | — | — |
| Doctor Interface | — (base MedGemma) | — | — |

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
