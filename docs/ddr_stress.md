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

### A. Chaos DRAM slave ✅ LIVRÉ (commits S1-S2, `ddrImpl2`)

- **Code** : `spinalML/test/src/spinalML/harness/DramChaos.scala` —
  `DramChaosConfig` (`disabled` / `light` / `heavy`, seeds déterministes) +
  `DramChaosInterposer` (interposeur RTL test-only entre `dut.io.axiMaster`
  et `AxiMemorySim) + `ChaosGate` (FIFO données + FIFO release-time, hold LFSR
  par transaction). Ordre strict par canal (jamais de réordonnancement intra-ID),
  blackouts refresh, pénalités row-miss/turnaround depuis l'historique AR.
- **Preuve** : `spinalML/test/src/spinalML/nn/DramChaosSpillTest.scala` —
  Linear spill P=2 sous chaos-heavy, oracle `ModelReplica` **inchangé** :
  - I8-K8-P2 : `dev=0.0` en 116 cycles (vs 93 idéal), AR=8 / AW=1 ;
  - BF16-K8-P2 : `dev=0.0` en 116 cycles, AR=14 / AW=1.
  Mêmes beats qu'en idéal (les gates préservent le trafic, seul le timing glisse),
  aucun deadlock. Leçons de bring-up : règle same-tag parent/enfant pour `<>`
  (le flip s'annule), ports top-level exposés (`ctrlBus`, `outStream`, `memPort`),
  sondes sur `memPort` (le cône interne `dut.io.axiMaster` est élagué par Verilator).
- **Reste (généralisation, commits T1-T3, `ddrImpl2`)** ✅ LIVRÉ :
  - `ChaosDut` trait (`harness/DramChaos.scala`) : le harness programme contre
    des accesseurs concrets (`memAxi`/`ctrlAxi`/`outShape`/`outLanes`/`outValid`/…)
    au lieu d'un type wrapper — zéro generics douloureux ;
  - `UniversalTestHarness.runStress` : même moteur bit-exact que `run` (corps
    partagé `execute`), timeout défaut relevé (200000) pour les runs ralentis ;
  - flag CLI `spinalml test --stress [--stress-level light|heavy] [--stress-seed N]` :
    le scaffold élabore un `StressWrapper` (DUT + interposeur + ports exposés),
    l'oracle tourne inchangé sur `wrapper.dut`, `runStress` partage le moteur ;
  - chaîne complète prouvée sur Linear : `spinalml test UniversalSpillDemo`
    (idéal, R4) + `spinalml test --stress [--stress-level light]` (chaos,
    bit-exact), non-régression `Universal1DDemo` sans stress.

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
2. **Chaos model (A)** ✅ livré et prouvé sur les suites spill (S1-S2 : 2/2,
   `dev=0.0`, pas de deadlock) ; généralisation harness/CLI `--stress` ensuite.
3. **Conv-spill** avec le gabarit Linear (knob `LayerSpec`, slice fetch `Sequential`,
   fold réplica, layout si l'ordre W change, e2e + formel).
4. **LiteDRAM sim (C)** comme backend, mêmes suites bit-exactes.
5. **Bring-up HW** : PHY, calibration, mesures (Phase 5b).

Le chaos model (2) profite directement à Conv-spill (3), et les deux dérisquent
LiteDRAM (4). Les ops pointwise (ReLU, pools, norms, activations) n'auront jamais
de spill : poids minuscules ou nuls, streaming pur — l'extension concerne en
pratique Linear (fait) → Conv → matmuls internes d'attention.
