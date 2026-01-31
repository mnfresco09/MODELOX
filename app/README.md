# 🖥️ MODELOX Web Dashboard

> **Módulo Opcional** - Dashboard web para visualización de resultados.

---

## ⚠️ Dependencias Separadas

Este módulo tiene sus propias dependencias que **NO** están incluidas en el requirements.txt principal del proyecto.

### Instalación Backend

```bash
cd app/backend
pip install -r requirements.txt
```

### Dependencias específicas:
- `fastapi` - Framework web async
- `uvicorn` - Servidor ASGI
- `pydantic` - Validación de datos
- `psutil` - Monitoreo del sistema

---

## 🚀 Ejecución

### Con Docker (recomendado):
```bash
cd app
docker-compose up --build
```

### Sin Docker:
```bash
# Terminal 1 - Backend
cd app/backend
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd app/frontend
npm install
npm run dev
```

---

## 📁 Estructura

```
app/
├── backend/          # FastAPI REST API
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/         # React + Vite
│   ├── src/
│   ├── package.json
│   └── Dockerfile
├── nginx/            # Reverse proxy config
└── docker-compose.yml
```

---

**NOTA:** Este módulo es completamente opcional y no afecta al funcionamiento del core de MODELOX.
