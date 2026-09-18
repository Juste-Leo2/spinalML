# Saturation MNIST RNE + DSP Gowin (Wave 5) — bug nextpnr « CIN from logic »

## 1. Symptôme

Sur Tang Primer 20K, l'auto-test MNIST déterministe (`examples/Mnist/inference.py`)
avec un bitstream **RNE + 25 `MULT18X18` explicites** renvoie des logits saturés :

```text
HW logits : -448.00 -448.00 +448.00 +448.00 -448.00 -448.00 -448.00 -448.00 -448.00 -448.00
```

alors que le même modèle construit avec `--rounding trunc`, avec `--no-dsp`, ou
au commit `7a96779` (Wave 4, RNE + DSP) est bit-exact vis-à-vis du réplica NumPy.

## 2. Matrice de reproduction

| Build | Commit | RNE/trunc | DSP | Résultat HW |
|---|---|---|---|---|
| `hw_build/bisect-rne/b823d07-rne` | `b823d07` | RNE | 25 | **sature ±448** |
| `hw_build/bisect-rne/b823d07-rne-nodsp` | `b823d07` | RNE | 0 | bit-exact |
| `hw_build/bisect-rne/b823d07-trunc` | `b823d07` | trunc | 25 | bit-exact |
| `hw_build/bisect-rne/7a96779` | `7a96779` | RNE | 25 | bit-exact |
| `hw_build/bisect-rne/b823d07-rne-fixed` | `b823d07` + backport | RNE | 25 | **bit-exact (fix)** |

Le commit `b823d07` (« Wave 5, step 1 ») n'ajoute que la propagation NaN
(`Float.mul/add/gt/widen/roundTo`) et le fil local `mantW` de
`roundTo` (widen) — mais le seul delta RTL entre les lanes trunc et RNE de ce
même commit est l'arrondi de `Float.fromSInt`.

## 3. Ce qui a été écarté

- **RTL/logique Scala** : le module `CastOp_1` généré (RNE et trunc) a été
  simulé exhaustivement sur les 65 536 entrées I16, en RTL et après
  synthèse générique : **0 mismatch** avec le golden Python.
- **Simple violation de setup globale** : chemin critique PnR
  `7a96779` 28.29 ns (OK) / `b823d07-rne` 28.50 ns (KO) /
  `b823d07-rne-nodsp` 28.11 ns (OK) / `b823d07-trunc` 25.45 ns (OK).
  Un écart de 0.2 ns ne peut pas expliquer un KO déterministe ; aucun Fmax
  reporté n'est sous 27 MHz.
- **Câblage des primitives** (`scripts/diff_hw_json.py`, rapports
  `hw_build/bisect-rne/diff/`) : sur les 3 paires KO/OK, les 25 `MULT18X18`
  ont **0 différence de connexions/paramètres** (synth et pnr). Les séquences
  de passes Yosys et les 19 warnings sont identiques. Les écarts mémoire
  (`DI->DO`) sont au même niveau de bruit que la paire de contrôle
  `trunc vs 7a96779` (deux builds qui marchent) : renommages Yosys, pas de
  signal fonctionnel.

## 4. Cause racine

Bug **nextpnr-himbaechel (arch Gowin)**, corrigé upstream par le commit
**`6030081a15`** (2026-09-13, *« gowin: fix carry-in adapter table for CIN from
logic »*) — absent du snapshot OSS-CAD-Suite du 2026-09-06
(`nextpnr 0.11.1-19-g8dbcee5c`).

Quand une carry-chain a son CIN qui vient de la logique, nextpnr insère un
« head ALU » qui prend ce signal sur `I0` et laisse `CIN` non connecté. L'ancien
`RAW_ALU_LUT = 0x505a` donne `COUT = I0 | CIN` : la retenue injectée dans la
chaîne **dépend alors de l'état de retenue de l'ALU voisin physiquement chaîné**
(placement). Le fix passe à `0x000a` (`COUT = I0`, CIN ignoré).

Le build RNE du cast `Float.fromSInt` introduit exactement des carry-from-logic
(round-up `mantissa +^ roundUp`, carry `expSInt + mantOv`). Sur les 11 head-ALUs
`CIN_NETTYPE=LOGIC` du design, celle du cast et/ou des accumulateurs se retrouve
chaînée sur une carry-chain active dans le build KO : les additions sont fausses,
les activations partent en saturation, les logits FP8 saturent à ±448. Les trois
builds de référence ont le même nombre de head-ALUs (10-11) mais un placement
qui ne les chaîne pas sur une carry active — d'où le caractère dépendant du
layout (et non du Scala).

## 5. Correctif

Backport au niveau `pnr.json`, avant `gowin_pack`, dans
`cli/spinalml_cli/build_runner.py::patch_gowin_cin_from_logic()` : les ALUs
`CIN_NETTYPE=LOGIC` avec `RAW_ALU_LUT` = `0x505a` passent à `0x000a`. Le routage
et les autres fuses sont inchangés (bitstream corrigé : 123 octets de différence
seulement, 11 ALUs × 4 fuses). Le correctif est appliqué automatiquement à
chaque build Gowin ; il devient un no-op quand nextpnr est mis à jour.

Bitstream RNE corrigé (généré depuis le `pnr.json` du build KO, sans autre
changement) : `hw_build/bisect-rne/b823d07-rne-fixed/top.fs`.

Contournements historiques conservés (utiles si le backport est retiré) : lane
RNE sans DSP matériels (validée, 0 DSP, 14 155 LUTs, Fmax 35.58 MHz) ou arrondi
trunc (25 DSP, 10 390 LUTs, Fmax 39.29 MHz).

## 6. Validation matérielle

Validé sur Tang Primer 20K le **2026-09-18** :

- flash de `b823d07-rne-fixed/top.fs` (même design/placement/routage que le
  build KO, seuls les 11 head-ALUs patchés — 123 octets de différence) :
  l'auto-test MNIST redevient bit-exact vs la réplique NumPy
  (`-7.50 -6.00 -5.00 -1.25 -3.25 -2.00 -3.50 -4.50 -3.00 -6.00`, classe 3) ;
- chaîne complète reconstruite avec le CLI (`build` → backport auto → flash →
  `inference.py`) : self-test OK. **Wave 5 validée en RNE + DSP matériels.**

## 7. Outil de diagnostic

`scripts/diff_hw_json.py` compare deux builds (synth.json / pnr.json /
build.log) et classe les écarts en fonctionnel / placement / routage. Le mode
matrice rejoue les 4 paires de référence :

```bash
python scripts/diff_hw_json.py --matrix \
    --out-dir hw_build/bisect-rne/diff
```

## 8. Suivi

- Le backport `patch_gowin_cin_from_logic()` peut être supprimé dès que
  `nextpnr-himbaechel` embarque `6030081a15` (aucun head-ALU concerné → no-op).
- **Dette technique assumée** : le backport vit dans
  `cli/spinalml_cli/build_runner.py` (à côté de `ensure_apycula_patched()`).
  À terme, regrouper ces rustines toolchain (nextpnr/apycula, `pnr.json`/package
  patches) dans un module dédié du CLI plutôt que dans le runner de build, pour
  que la suppression du backport soit un simple retrait de fichier.
- Référence upstream : <https://github.com/YosysHQ/nextpnr/commit/6030081a15>
- Piste d'outillage complémentaire (non bloquante) : la base de timing du
  chipdb GW2A n'a pas d'entrée DSP (`lut, alu, sram, dff, bram, ...`), donc les
  chemins internes aux `MULT18X18` ne sont pas analysés par la STA.
