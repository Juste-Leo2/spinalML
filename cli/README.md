# SpinalML CLI

L'outil en ligne de commande (CLI) de **SpinalML** permet d'orchestrer facilement toute la chaîne de conception FPGA (compilation SpinalHDL, simulations Verilator, synthèse Yosys, placement-routage NextPNR, programmation hardware openFPGALoader) sans exiger d'installation système complexe.

> [!IMPORTANT]
> **Règle d'or :** Que vous utilisiez les sources Python (`python cli/main.py`) ou le binaire autonome (`./spinalml`), les commandes doivent **toujours être exécutées depuis la racine du projet** (`spinalML/`).

---

## 1. Binaires Précompilés Autonomes (Recommandé)

Des exécutables autonomes uniques (single-file) sont automatiquement générés et disponibles au téléchargement sur la page des [**GitHub Releases**](https://github.com/Juste-Leo2/spinalML/releases) :

| Plateforme | Archive Release | Binaire exécutable |
| :--- | :--- | :--- |
| **Windows (x64)** | `spinalml-windows-x64.zip` | `spinalml-windows-x64.exe` |
| **Linux (x64)** | `spinalml-linux-x64.tar.gz` | `spinalml-linux-x64` |
| **Linux (ARM64)** | `spinalml-linux-arm64.tar.gz` | `spinalml-linux-arm64` |
| **macOS (Apple Silicon)** | `spinalml-darwin-arm64.tar.gz` | `spinalml-darwin-arm64` |
| **macOS (Intel)** | `spinalml-darwin-x64.tar.gz` | `spinalml-darwin-x64` |

### Que contient le binaire ?
- **Ultra-léger (~15 Mo)** : Le binaire embarque le moteur CLI pur (`typer`, `rich`, `requests`), les profils de cartes FPGA (`boards/`), le framework matériel (`spinalML/`) et la configuration de build (`build.mill`).
- **Ce qui n'est PAS embarqué** : Les tests de co-simulation Python (`cocotb`, `pytest`). Cocotb nécessite une compilation VPI dynamique liée au CPython 3.12 local du système (sur Linux ou WSL). Les utilisateurs souhaitant exécuter la suite de tests Python de développement doivent utiliser l'environnement source complet (voir [`docs/building_from_source.md`](../docs/building_from_source.md)).

### Démarrage rapide avec le binaire :
```bash
# 1. Téléchargez et décompressez l'exécutable pour votre OS depuis les Releases
# 2. Placez-vous à la racine de votre projet SpinalML
cd spinalML

# 3. Initialisez les outils EDA (OSS CAD Suite, Mill, w64devkit)
./spinalml setup

# 4. Compilez un modèle Scala en Verilog
./spinalml compile tests/universal/UniversalOpsDemo.scala -o rtl/

# 5. Lancez la vérification bit-exacte sous Verilator
./spinalml test tests/universal/UniversalOpsDemo.scala
```

---

## 2. Reproduction des Binaires depuis les Sources

Pour compiler vous-même le binaire autonome avec PyInstaller :

```bash
# Depuis la racine du projet spinalML :
uv pip install -r cli/requirements.txt pyinstaller

# Lancer le script de compilation
python cli/build_binary.py
```
Le binaire résultant sera généré dans le dossier `dist/`.

---

## 3. Utilisation depuis les Sources Python

Si vous préférez exécuter le CLI sans compiler de binaire :

1. **Installer les dépendances minimales :**
   ```bash
   uv pip install -r cli/requirements.txt
   ```
2. **Invoquer le CLI via `python cli/main.py` :**
   ```bash
   python cli/main.py --help
   ```

---

## 4. Commandes Principales

| Commande | Description |
| :--- | :--- |
| `spinalml setup` | Télécharge et configure automatiquement Mill et OSS CAD Suite dans `~/.spinalml_tools` |
| `spinalml compile <modele.scala>` | Compile un modèle Scala en Verilog autonome et génère le wrapper UartSoC |
| `spinalml test <modele.scala>` | Exécute la vérification universelle bit-exacte sous simulation C++ Verilator |
| `spinalml build <modele.scala>` | Synthèse Yosys -> Placement & Routage NextPNR -> Génération du bitstream `.fs` |
| `spinalml flash` | Programme la carte FPGA connectée en USB via openFPGALoader |
| `spinalml mill [...]` | Wrapper direct vers l'outil de build Mill Scala |
| `spinalml verilator [...]` | Wrapper direct vers le simulateur Verilator |
| `spinalml yosys [...]` | Wrapper direct vers l'outil de synthèse Yosys |
| `spinalml nextpnr <arch> [...]` | Wrapper direct vers NextPNR (`ice40`, `himbaechel`...) |
| `spinalml openfpgaloader [...]` | Wrapper direct vers l'utilitaire de programmation JTAG/UART |
