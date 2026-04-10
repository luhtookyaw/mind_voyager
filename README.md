# MindVoyager Client Simulator

This repository includes a small Python implementation of the client simulator
described in `MindVoyager.pdf`, wired to the local Patient-PSi-CM planning and
resistance dataset.

## Run a dry run

```bash
python3 -m mind_voyager.client_simulator --case-id 1-1 --difficulty hard --dry-run
```

The dry run prints the therapist intake plus the currently visible, masked
client prompt. It does not call an LLM.

## Run an interactive simulator

```bash
export OPENAI_API_KEY=...
python3 -m mind_voyager.client_simulator --case-id 1-1 --difficulty normal
```

The default dataset path is:

```text
data/Patient_Psi_CM_Dataset.json
```

The default model is `gpt-4o-mini`; pass `--model` to change it.

## Difficulty Mapping

The simulator follows the paper's openness and metacognition setup:

| Difficulty | Initial visible external elements | Metacognition interval |
| --- | ---: | ---: |
| easy | 3 | 1 |
| normal | 2 | 2 |
| hard | 1 | 3 |

External diagram elements are situation, automatic thought, emotion, and
behavior. Internal diagram elements are history, core beliefs, intermediate
belief, and coping strategies. This local dataset does not include the paper's
resistance-level field, so difficulty is selected explicitly with
`--difficulty easy|normal|hard`.
