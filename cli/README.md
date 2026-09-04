# SpinalML CLI

Ce dossier contient l'outil en ligne de commande (CLI) Python permettant de configurer et d'utiliser facilement la chaîne d'outils FPGA (Mill, Verilator, Yosys, NextPNR, etc.) nécessaire au projet `spinalML`, sans exiger d'installation système globale.

## Prérequis

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (recommandé pour la gestion d'environnement)

## Installation de l'environnement CLI

1. **Créer l'environnement virtuel avec uv :**
   ```bash
   cd cli
   uv venv
   ```
2. **Activer l'environnement virtuel :**
   - Sur Windows : `.venv\Scripts\activate`
   - Sur Linux/macOS : `source .venv/bin/activate`
3. **Installer les dépendances :**
   ```bash
   uv pip install -r requirements.txt
   ```

## Utilisation

Le CLI est invoqué via `main.py`. 

### Initialisation

Télécharge et extrait les outils nécessaires (OSS CAD Suite, Mill) dans `~/.spinalml_tools`.

```bash
python main.py setup
```

### Commandes disponibles

Les commandes suivantes redirigent l'exécution vers les binaires installés localement :

- `python main.py mill [args]` : Lance Mill (ex: `python main.py mill version`)
- `python main.py verilator [args]` : Lance Verilator
- `python main.py sby [args]` : Lance SymbiYosys
- `python main.py nextpnr [args]` : Lance NextPNR (nextpnr-ice40, nextpnr-ecp5, etc. selon les arguments)
