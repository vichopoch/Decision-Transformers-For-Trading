# agents.md
## Contexto del proyecto
Estás operando sobre un fork de **ra9hur/Decision-Transformers-For-Trading**.  
El objetivo es convertirlo en un sistema **Elastic-Evolve Trader** capaz de:
1. Entrenar un **Elastic Decision Transformer (EDT)** para formar y reequilibrar una cartera de acciones del **S&P IPSA**.
2. Ejecutar un bucle evolutivo AlphaEvolve-style donde **o3-high** genera parches de código (diffs) que mejoran reward, features y arquitectura.
3. Funcionamiento 100 % local en un **MacBook Pro M1 Max (32 GB RAM)** usando GPU MPS.

---

## Metas de alto nivel
| ID  | Descripción | Éxito cuando… |
|-----|-------------|---------------|
| G1  | Reemplazar dataset y entorno por acciones IPSA (OHLCV + factores) | `pytest tests/test_env.py::test_ipsa_shape` aprueba |
| G2  | Sustituir Decision Transformer → **Elastic DT** (`models/edt.py`) con context lengths `[1,5,10,20,60]` | Script `train_edt.py --verify` genera checkpoint y métricas |
| G3  | Crear módulo `evolve/alpha_loop.py` que: <br>• marque bloques mutables <br>• invoque OpenAI o3-high <br>• aplique diffs <br>• evalúe con Ray & vectorbt | `make mutate N=10` produce `/artifacts/champion.json` |
| G4  | Añadir scripts CLI (`train.py`, `mutate.py`, `deploy.py`) y documentación | `python -m elastic_trader.deploy --date YYYY-MM-DD` crea CSV de órdenes |
| G5  | Mantener compatibilidad MPS/CPU y pasar `pylint` + `mypy` sin errores | `make check` termina con status 0 |

---

## Roadmap detallado

### 1. Datos y entorno
- **1.1 Descargar lista IPSA**: `utils/data_loader.py` debe usar el endpoint *S&P IPSA* o fallback a scraping simple.  
- **1.2 OHLCV diario**: implementar en `data/download_data.py` con `yfinance`, ajustado a huso CLP.  
- **1.3 Entorno Gym**: clonar `finrl.meta.env_stock_trading` y renombrar a `envs/ipsa_env.py`; agregar coste de transacción 0.1 %.  
- **1.4 Features**: pipeline en `features/feature_pipeline.py` (RSI, MACD, UF, USD/CLP, cobre).

### 2. Elastic Decision Transformer
- Copiar base de **Elastic-DT** (`kristery/Elastic-DT`) a `models/edt.py`; adaptar a PyTorch 2.2 y MPS.  
- Parametrizar `context_lengths` y exponerlos a la capa evolutiva.

### 3. Bucle evolutivo con o3-high
- `evolve/alpha_loop.py`  
  - **EVOLVE-BLOCKS**: feature engineering, reward shaping, hparams, context grid.  
  - Prompt builder para o3-high (modelo `o3-high`).  
  - Aplicar unified diff a código; re-entrenar en Ray (8 workers).  
  - Selección Pareto (Sharpe, Sortino, MDD, turnover).  
  - Guardar artefactos en `minio://elastic-evolve`.

### 4. CLI y despliegue
- `scripts/train.py` ➜ entrenamiento baseline.  
- `scripts/mutate.py` ➜ Lanza `N` ciclos del bucle evolutivo.  
- `scripts/deploy.py` ➜ Infiere pesos diarios y guarda `orders/YYYY-MM-DD.csv`.

### 5. Tests y CI
- Carpeta `tests/` con PyTest: formas de tensores, reward consistency, compilación de parches.  
- GitHub Action que ejecuta `make lint test`.

### 6. Compatibilidad Mac M1 Max
- En `environment.yml` fija `torch==2.2.*`, `accelerate`, `vectorbt` y habilita device `"mps"`; fallback a CPU.  
- Asegura `torch.set_default_dtype(torch.float32)` para evitar errores de precisión en MPS.

---

## Estándares de código
- **Licencia**: mantener MIT.  
- **Sin placeholders**: cada función debe estar implementada por completo.  
- **Estilo**: PEP8, docstrings NumPy-style, tipado `typing`.

---

## Comandos rápidos (`Makefile`)

```make
make download     # Descarga datos IPSA
make train        # Entrena EDT baseline
make mutate N=20  # Corre 20 mutaciones AlphaEvolve
make deploy DATE=2025-05-22  # Genera órdenes para el día T
make check        # pylint + mypy + pytest
