# Budget de tests — durées mesurées (août 2026)

> **Pourquoi ce fichier** : la porte de régression complète (SMN + Phase 3) dépasse 15-25 min
> wall, dominée par les complètes simulations 64×64. Connaître la hiérarchie de coût permet de
> choisir l'étage de vérification adapté à un changement sans « relancer 20 minutes à l'aveugle ».
>
> **Règle d'usage** : les suites rapides/moyennes = la boucle de dev quotidienne (une seule
> invocation mill, jamais en parallèle — voir leçon M1.7) ; les suites lourdes = à lancer
> explicitement (ou en CI nightly), jamais enchaînées à la va-vite.

## Durées mesurées (ms, sauf N/A pour compilation/validation sans mesure)

| Suite de tests | Nom du test | Durée (ms) |
|----------------|-------------|------------|
| **DagTopologyTest** | ResidualMLPTemplate compilation | ~1918 |
| | Concat DAG compilation | ~485 |
| | DAG validation: forward reference rejected | N/A |
| | DAG validation: dtype mismatch rejected | N/A |
| | DAG validation: shape mismatch on Add rejected | N/A |
| | Residual MLP SoC runtime golden (skip connection, BF16 [2,4]) | 78.238 |
| **WideConvTilingTest** | WideConv tileHeight=64 | 271114.786 |
| | WideConv tileHeight=16 | 268577.907 |
| **WideResidualTilingTest** | WideResidual PLAIN chain (convK1, no fork) tiled vs replica | 327661.603 (2 sims) |
| | WideResidual SKIP chain (tap fork) tiled vs replica — 16×16 (`S4_GATE=1 WIDE_SIDE=16 WIDE_TILES=16`) | ~30 s |
| | WideResidual SKIP chain tiled vs replica — 64×64 (`WIDE_SIDE=64 WIDE_TILES="64,16"`) | 3192085.918 (53 min 12 s, run utilisateur) |
| **MnistChainedTest** | Mnistw4a8: chained in one session | 17120.528 |
| | Mnist BF16: chained in one session | 14758.030 |
| **MnistContinuousTest** | Mnist BF16: continuous RUN auto-advance | 24237.493 |
| | Mnistw4a8: continuous RUN auto-advance | 11026.895 |
| **WeightPrefetchChainTest** | Mnist BF16: eager weight prefetch | 25311.371 |
| | Mnistw4a8: eager weight prefetch | 10268.165 |
| **Im2ColContinuityTest** | Im2Col: band-seam stall equivalence | 454.896 (2 sims) |
| | Im2Col: two separate commands (M2.2) | 568.021 (3 sims) |
| **WeightResidentChainTest** | Mnist BF16: residency | 34071.400 |
| | Mnistw4a8: residency | 13651.030 |
| **BandTilingTest** | Mnist BF16: banded tileHeight variants | 44329.442 (3 sims) |
| | Mnistw4a8: banded tileHeight variants | 17638.526 (3 sims) |
| **RepackTest** | streaming repack 2→4 | 4.847 |
| | compilation-only (I8/FP8/I16/BF16) | N/A |
| **DoubleBufferStreamerTest** | generate_verilog | N/A |
| **StreamDoubleBufferTest** | Ping-Pong logic | 5.331 |
| **RepackStallDiffTest** | M1-A split BF16 4→1 random stalls | 44.237 |
| | M1-A split BF16 4→1 hot producer | 46.449 |
| | M1-A split I8 4→1 random stalls | 18.801 |
| | M1-A split I8 16→4 random stalls | 13.965 |
| | M1-A aggregate I8 2→4 random stalls | 7.914 |
| | M1-A aggregate I8 2→4 hot producer | 8.319 |
| | M1-C aggregate I8 1→4 random stalls | 13.002 (2 sims) |
| | M1-A chain I8 16-1-25 random stalls | 29.845 |
| | M1-A chain I8 16-1-25 slow consumer | 21.254 |
| | M1-A cross-check legacy adapter | 34.084 (2 sims) |

**Total mesuré : 40+ tests verts** (dont le SKIP gate de WideResidual — voir hiérarchie ; les tests longs `spinalML.heavy.*` et `S4_GATE=1` restent hors ci-simulations).

## Hiérarchie de coût

| Étage | Suites | Budget |
|---|---|---|
| **Tier 1 — rapide** (< 2 min) | RepackTest, StreamDoubleBufferTest, DoubleBufferStreamerTest, Im2ColContinuityTest, DagTopologyTest (runtime golden ~78 ms), RepackStallDiffTest | ~10 s cumulés |
| **Tier 2 — moyen** (10 s → 45 s/test) | MnistChainedTest, MnistContinuousTest, WeightResidentChainTest, WeightPrefetchChainTest, BandTilingTest | ~3-4 min (les deux modèles) |
| **Tier 3 — lourd** (5-11 min/test) | `spinalML.heavy.*` : WideConvTilingTest, WideResidualTilingTest (PLAIN) | ~20+ min (dominé par compile Verilator ~2-3 min × variants + sim 64×64) |
| **SKIP gate** (`S4_GATE=1`) | WideResidualTilingTest — PLAIN + SKIP tiled | 16×16 : ~1 min · 64×64 : ~55 min — à lancer explicitement en local |

> **Exclusion CI** : les suites Tier 3 vivent dans `spinalML/test/src/spinalML/heavy/` (et les
> modèles `WideConv`/`WideResidual`/`WideResidualPlainChain` dans `spinalML/src/spinalML/heavy/`).
> `ci-simulations.yml` balaie seulement les dossiers `ops/layers/examples/...` → le runner Radxa
> **ne compile ni n'exécute jamais** les Wide* : ils ne tournent que sur machine locale, en
> commande explicite. Ne pas remonter ces fichiers vers `examples/`.

## Knobs d'accélération existants

| Variable | Effet | Usage |
|---|---|---|
| `W4A8_ONLY=1` | skip les corps BF16 (affichés CANCELED — normal) | boucle rapide |
| `MNIST_TIMEOUT=<cycles>` | borne d'attente d'un passage (défaut variable : 5 M pour Mnist*, 800 k pour Wide*) | échec rapide / CI budget |
| `WIDE_TILES="64"` | sous-ensemble des tileHeights (WideResidualTilingTest) | T3 hors nightly |
| `MNIST_CONT_N`, `MNIST_CHAIN_N`, `MNIST_CHAIN_SEED`, `MNIST_PREFETCH_SEED` | réduction/sélection des itérations et images | ciblage |
| `S4_GATE=1` | active le SKIP gate de WideResidual (✅ VERT depuis M3.5 — fix push FIFO non gaté) | porte S4 : doit être **bit-exact** aux petites et pleines tailles |
| `SML_DEBUG=1` | logs `[DEBUG]` (tables nœuds, détails par couche, équilibre des flux) | diagnostic — défaut : `[INFO]` seul |
| `SML_DEBUG=2` | logs `[TRACE]` (beats/fires, spine back-pressure, per-frame) — flot massif | mesure ciblée (pipe vers fichier) |
| `SML_DEBUG_TAG=TAP` | restreint DEBUG/TRACE à un seul tag, ex. `TAP`, `ENGINE`, `WIDE` | TRACE focusée sans noyer la sortie |

## Logs SimLog (niveaux, tags)

Tous les tests/benches passent par `spinalML/utils/SimLog.scala` — format `[LEVEL] [TAG] msg` :

| Niveau | Déchencheur | Usage |
|---|---|---|
| `[ERROR]` `/` `[WARN]` `/` `[INFO]` | toujours | verdicts, exclusions de knobs, audit modèle, produits des probes |
| `[DEBUG]` | `SML_DEBUG=1` | tables complètes nœuds, valeurs (décimal + bits ≤8b), counts/stalls |
| `[TRACE]` | `SML_DEBUG=2` | per-beat / per-fire / per-frame |

Tags standards : `[DAG] [TAP] [SIM] [MODEL] [WIDE] [MNIST] [FORMAL]` + tags de probes `[FORK] [K1] [IM2COL]`.
Chaque suite renomme ses prints en `SimLog.info/debug(…)` — **aucun `println` nouveau**.

> **V2 logs (à venir, PAS dans la V1)** : observation en direct de tous les tenseurs internes du
> `Sequential` complet (ports de diagnostic `diag` optionnels) — aujourd'hui les suites haut
> niveau (import du modèle complet) ne voient que l'entrée et la sortie ; les valeurs de couches
> internes viennent des probes op-nus (DAG construit à plat dans les tests).

## Commandes de référence

```bash
# Tier 1+2 (boucle quotidienne, une seule invocation)
./mill spinalML.test.testOnly \
  spinalML.examples.MnistChainedTest \
  spinalML.nn.DagTopologyTest \
  spinalML.memory.RepackStallDiffTest \
  spinalML.ops.RepackTest \
  spinalML.memory.StreamDoubleBufferTest \
  spinalML.memory.DoubleBufferStreamerTest \
  spinalML.examples.WeightResidentChainTest \
  spinalML.examples.WeightPrefetchChainTest \
  spinalML.examples.MnistContinuousTest \
  spinalML.examples.Im2ColContinuityTest \
  spinalML.examples.BandTilingTest

# Tier 3 (à part, jamais sur le runner Radxa)
./mill spinalML.test.testOnly spinalML.heavy.WideConvTilingTest
./mill spinalML.test.testOnly spinalML.heavy.WideResidualTilingTest                     # PLAIN

# SKIP gate (S4) : débogage rapide 16×16 puis certification 64×64 (~55 min)
S4_GATE=1 WIDE_SIDE=16 WIDE_TILES=16 ./mill spinalML.test.testOnly spinalML.heavy.WideResidualTilingTest
S4_GATE=1 WIDE_SIDE=64 WIDE_TILES=64,16 ./mill spinalML.test.testOnly spinalML.heavy.WideResidualTilingTest

# Probes bruts (chaîne skip composant-exacte + conv K=1/K=3 nu) — boucle S4, < 15 s
./mill spinalML.test.testOnly spinalML.examples.ForkChainCountTest spinalML.examples.K1ConvCountTest

# Robustesse multi-seeds du préfetch (cause racine réglée — à relancer si réouverture)
for s in 1502108522 1666584426 345748929 747740128; do
  SPINAL_SIM_SEED=$s W4A8_ONLY=1 MNIST_PREFETCH_SEED=$s \
    ./mill spinalML.test.testOnly spinalML.examples.WeightPrefetchChainTest
done

# Formels (lourds, ~1-2 min chacun)
PATH="$HOME/.local/bin:$PATH" ./mill spinalML.test.runMain spinalML.symbolicTest.memory.StreamDoubleBufferHoldFormal
PATH="$HOME/.local/bin:$PATH" ./mill spinalML.test.runMain spinalML.symbolicTest.memory.StreamDoubleBufferFormal
```

## Notes
- Les durées Tier 3 dominent par le **compile Verilator par variante** (un `compile` par tileHeight) + la sim 64×64 (4 images × 2 variants ≈ 11 min) — réduction possible via `WIDE_TILES` ou en réduisant les images (1 au lieu de 3).
- Le **SKIP gate** de `WideResidualTilingTest` est actif derrière `S4_GATE=1` et **vert** depuis M3.5 (cause racine = push FIFO du TapBuffer non gaté sur le handshake du tee — voir `open-mysteries.md` M3.5). La certification 64×64 (WIDE_TILES="64,16") dépasse largement l'heure : privilégier le 16×16 pour la boucle.
- **Nouveaux probes S4** : `ForkChainCountTest` (chaîne du skip au niveau composant, compteurs + séquences de valeurs + vérif des paires du Add) et `K1ConvCountTest` (conv K=1 et K=3 nus, comptage exact) — les deux < 15 s, dans `spinalML/test/src/spinalML/examples/`.
