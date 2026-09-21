# DDR Stress — Stratégie de robustesse avant LiteDRAM

> **Statut** : doctrine figée le 21/09/2026 (discussion de cadrage post-PR « réplica spill »).
> **Docs liés** : `docs/ddr_final_impl.md` §S3 (chaîne validée sur Linear),
> `docs/ddr_replica_status.md` (contrat numérique, seam `--stress`),
> `docs/ddr_impl.md` Phase 5 (bring-up).
> **Décision structurante** : on valide la chaîne complète sur **Linear** d'abord
> (PR réplica spill, commits R1-R6), la **Conv** suit avec le même gabarit une
> fois Linear verrouillé. LiteDRAM remplacera l'IP propriétaire Gowin le moment venu.

## 0. Pourquoi un stress DDR : BRAM ≠ DDR

- **BRAM** (`spinalML/src/spinalML/memory/BramAdapter.scala:85-86`) : `readSync`,
  latence fixe d'1 cycle, déterministe. La sim actuelle est (quasi) cycle-exacte.
- **Sim actuelle** (`AxiMemorySim`, modèle idéal) : réponses immédiates, cohérence
  séquentielle. Elle prouve la correction fonctionnelle + la conformité protocole,
  **pas** le comportement sous timing DRAM réel.
- **Vraie DDR** : latence variable — refresh périodique, row hit/miss, conflits de
  banques, turnaround read/write, arbitrage du contrôleur. Rien de tout ça n'existe
  dans nos sims aujourd'hui.

Nos preuves e2e bit-exactes reposent donc sur un modèle mémoire gentil.
Le stress bouche ce trou **avant** LiteDRAM, pas après.

## 1. Ce qui est modélisable en sim (le « gros stress test »)

Le design aide : tout circule en `Stream(valid/ready)` élastique, les DMAs comptent
des beats, pas des cycles — le design est latency-insensible par construction, le
stress sert à le **prouver**. Côté ordre : `DMAReader.scala:102` force `ar.id := 0`
et consomme R comme un flux ordonné ; AXI garantit l'ordre par ID même sur un vrai
contrôleur, donc le seul réordonnancement légal est inter-ID au niveau de l'arbiter
(schéma route-bit de `Accelerator`) — modélisable par entrelacement inter-masters.

### A. Chaos DRAM slave (sim-only, branché sur le seam R4)

Remplace `AxiMemorySim` via `UniversalTestHarness.run(..., memorySimConfig = ...)`
(seam ajouté en R4, défaut = modèle idéal historique) :

1. latence de lecture **distributionnelle**, fonction de l'historique d'adresses
   (pénalités row-miss / bank-conflict simplifiées) ;
2. blackouts **refresh** périodiques + pénalité de **turnaround** R/W ;
3. jitter `ready/valid` sur AR/R/AW/W/B, `maxOutstandingReads` réduit ;
4. backpressure `outStream`, timing START/reset ;
5. réordonnancement **légal uniquement** (jamais intra-ID) ;
6. seeds déterministes, niveaux `light` / `heavy`.

Invariants exigés sous stress : `dev == 0.0`, aucun timeout/deadlock, fencing correct.
À appliquer **en premier aux suites spill** (multi-passes, RMW, `WaitFence` — le plus
timing-sensible : `SequentialSpillTest`, `SequentialReplicaSpillTest`, `MatmulSpillTest`),
puis à l'e2e replica et à la non-régression.

### B. Formels d'invariants (indépendants du timing)

Fencing, exactly-once, pas de drop, pas de deadlock sous inputs fair.
Valables quel que soit le backend mémoire — donc valables aussi pour LiteDRAM.
Référence : `SpillPassControllerFormal`, `SpillPassControllerLivenessFormal`,
`AcceleratorFormal`, `MemorySpecFormal`, `DMAWriterFormal`, `BiasAddFormal`.

### C. LiteDRAM comme backend sim

`ExternalDram → DdrAdapter → LiteDRAM` (modèle sim), mêmes suites bit-exactes :
le contrôleur réel remplace le chaos. Si A+B sont verts, l'intégration LiteDRAM ne
doit être qu'une question de timing, plus de correction.

## 2. Ce que LiteDRAM gère déjà / ce qui reste au hardware

**Non modélisable en sim fonctionnelle (phase HW, `ddr_impl.md` Phase 5b)** :
pads/PHY, intégrité signal, calibration ( leveling, training), timing closure,
bugs internes du contrôleur (responsabilité upstream LiteDRAM, déjà vérifié par eux).

Ces couches n'affectent pas la correction fonctionnelle **si** le contrôleur est
AXI-compliant — elles produisent les chiffres bande passante / latence / LUT / Fmax
au bring-up. D'où la répartition :

| Couche | Couverture |
|---|---|
| Correction fonctionnelle (ordre, fences, protocole) | A + B, 100 % en sim |
| Perf réelle (BW, latence, surface, Fmax) | C puis bring-up HW |
| PHY / calibration / signal | HW uniquement |

## 3. Règle d'or : le réplica ne connaît jamais le timing

**Le réplica est un oracle purement fonctionnel.** Aucun paramètre timing/pression
n'y entre, jamais — ni `memorySimConfig`, ni jitter, ni seed de stress. Le stress vit
exclusivement dans le chemin harness/CLI (timing), l'oracle reste bit-à-bit identique.

Concrètement, tout changement réplica suit la discipline appliquée en R1-R2 :
paramètre optionnel à **défaut = comportement historique au bit près**, jamais de
réécriture de l'ordre existant ; tout mismatch couche/layout est **rejeté** (pas de
double transposition silencieuse). Le futur `--stress` ne touchera que
`UniversalTestHarness` (config du modèle mémoire) et le scaffold CLI (passthrough
d'arguments) — cf. seam documenté dans `docs/ddr_replica_status.md`.

## 4. Ordre de marche acté

1. **Chaîne Linear verrouillée** ✅ (R1-R6 : layout, fold, e2e, CLI, trafic, gates).
2. **Chaos model (A)** sur les suites spill existantes — dérisque logique à peu de frais.
3. **Conv-spill** avec le gabarit Linear (knob `LayerSpec`, slice fetch `Sequential`,
   fold réplica, layout si l'ordre W change, e2e + formel).
4. **LiteDRAM sim (C)** comme backend, mêmes suites bit-exactes.
5. **Bring-up HW** : PHY, calibration, mesures (Phase 5b).

Le chaos model (2) profite directement à Conv-spill (3), et les deux dérisquent
LiteDRAM (4). Les ops pointwise (ReLU, pools, norms, activations) n'auront jamais
de spill : poids minuscules ou nuls, streaming pur — l'extension concerne en
pratique Linear (fait) → Conv → matmuls internes d'attention.
