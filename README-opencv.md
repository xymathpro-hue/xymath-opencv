# xyMath OpenCV API

API de correção automática de gabaritos usando OpenCV.

## Funcionalidades

- ✅ Detecção de marcadores de canto
- ✅ Correção de perspectiva
- ✅ Leitura de QR Code
- ✅ Detecção de bolhas preenchidas
- ✅ Processamento em lote

## Endpoints

### `GET /`
Health check - verifica se a API está online.

### `POST /api/process`
Processa uma imagem de gabarito.

**Request:**
```json
{
  "image": "base64_encoded_image"
}
```

**Response:**
```json
{
  "success": true,
  "qr_data": {
    "simulado_id": "abc123",
    "aluno_id": "def456",
    "turma_id": "ghi789",
    "total_questoes": 20
  },
  "answers": ["A", "B", "C", "D", null, "A", ...],
  "total_detected": 18,
  "total_questions": 20
}
```

### `POST /api/detect-qr`
Detecta apenas o QR Code (validação rápida).

### `POST /api/batch`
Processa múltiplas imagens de uma vez.

**Request:**
```json
{
  "images": ["base64_1", "base64_2", ...]
}
```

## Deploy no Railway

### 1. Criar conta no Railway
- Acesse https://railway.app
- Faça login com GitHub

### 2. Criar novo projeto
- Clique em "New Project"
- Selecione "Deploy from GitHub repo"
- Conecte o repositório

### 3. Configurar variáveis (opcional)
Não são necessárias variáveis de ambiente.

### 4. Deploy automático
O Railway fará o deploy automaticamente a cada push.

## Deploy Manual

```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Login
railway login

# Criar projeto
railway init

# Deploy
railway up
```

## Desenvolvimento Local

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Instalar zbar (Linux)
sudo apt-get install libzbar0

# Rodar servidor
python app.py
```

## Formato da Folha de Respostas

A folha de respostas deve ter:

1. **4 marcadores de canto** - Quadrados pretos nos 4 cantos
2. **QR Code** - Contendo: `simulado_id|aluno_id|turma_id|total_questoes`
3. **Grade de bolhas** - Círculos para A, B, C, D, E

```
■─────────────────────────────────────────────────────────■
│                                                         │
│   [QR CODE]     Nome: _______________________           │
│                                                         │
│   01 ○ ○ ○ ○ ○      11 ○ ○ ○ ○ ○                       │
│   02 ○ ○ ○ ○ ○      12 ○ ○ ○ ○ ○                       │
│   03 ○ ○ ○ ○ ○      13 ○ ○ ○ ○ ○                       │
│   ...                                                   │
│                                                         │
■─────────────────────────────────────────────────────────■
```

## Custo

**R$ 0,00** - Hospedagem gratuita no Railway (500 horas/mês)

## Tecnologias

- Python 3.11
- Flask
- OpenCV
- pyzbar (leitura de QR Code)
- Gunicorn
