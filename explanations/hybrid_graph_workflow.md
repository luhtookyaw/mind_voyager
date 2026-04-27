Use a 5-stage pipeline.

**1. Post ingestion**
Input:
- Reddit post text
- optional title, subreddit, comments

Store:
- `source_id`
- `source_text`
- `source_metadata`
- `retrieval_timestamp`

Do not build a case directly from raw text.

**2. External extraction**
Extract only what the post actually supports:

- `situation`
- `automatic_thought`
- `emotion`
- `behavior`

Also extract:
- topic tags
- uncertainty per field
- evidence spans from the post

Output schema:
```json
{
  "source_id": "reddit_123",
  "situation": "...",
  "automatic_thought": "...",
  "emotion": ["..."],
  "behavior": "...",
  "topics": ["family_conflict", "rejection", "avoidance"],
  "evidence": {
    "situation": "...",
    "automatic_thought": "...",
    "emotion": "...",
    "behavior": "..."
  },
  "confidence": {
    "situation": 0.92,
    "automatic_thought": 0.71,
    "emotion": 0.88,
    "behavior": 0.75
  }
}
```

This stage should not infer core beliefs yet.

**3. Graph retrieval**
Use the extracted external signals to retrieve from the hybrid graph.

Query with:
- situation embedding / keywords
- thought content
- emotion cluster
- behavior pattern
- topic tags

Retrieve from 3 layers:
- external statement nodes
- internal statement nodes
- abstract cluster nodes

Target outputs:
- top matching external episodes from Patient-psi-like cases
- top candidate core belief clusters
- top candidate intermediate beliefs
- top candidate coping strategies
- top candidate history patterns

Important:
- retrieve `k` candidates, not one
- keep scores
- keep provenance

Example retrieval result:
```json
{
  "candidate_internal_profiles": [
    {
      "core_beliefs": ["I am unlovable.", "I am trapped."],
      "intermediate_beliefs": ["If I show my true self, people will reject me."],
      "coping_strategies": ["avoidance of family conflict"],
      "history_patterns": ["maternal criticism and rejection"],
      "support_score": 0.81
    }
  ]
}
```

**4. Case construction**
Now generate a new case using:
- post-grounded external episode
- retrieved internal candidates
- dataset style constraints

Construction rule:
- external episode should stay close to the post
- internal structure should come from retrieved graph patterns
- final case should match Patient-psi format

Target output:
```json
{
  "name": "...",
  "type": ["..."],
  "history": "...",
  "helpless_belief": ["..."],
  "unlovable_belief": ["..."],
  "worthless_belief": ["..."],
  "intermediate_belief": "...",
  "intermediate_belief_depression": "...",
  "coping_strategies": "...",
  "situation": "...",
  "auto_thought": "...",
  "emotion": ["..."],
  "behavior": "..."
}
```

Generation prompt should say:
- preserve evidence from the post
- infer only plausible internal structure
- do not overfill unsupported fields
- use Patient-psi wording style
- if support is weak, mark a field as lower-confidence or leave it sparse

**5. Validation**
This is mandatory.

Run a validator on the constructed case:

Checks:
- does `situation -> automatic_thought` make sense?
- does `automatic_thought -> emotion` make sense?
- does `automatic_thought/coping -> behavior` make sense?
- do core beliefs plausibly support the intermediate belief?
- does history plausibly support the core beliefs?
- is the case too similar to an existing dataset case?
- is any field unsupported by the source post plus retrieved context?

Use:
- rule checks
- LLM consistency judge
- similarity filter against original dataset

Only keep cases above threshold.

**Graph requirements**
For this pipeline, your graph should contain:

Dataset-grounded nodes:
- history statements
- core belief statements
- intermediate belief statements
- coping statements
- situation statements
- automatic thought statements
- emotion statements
- behavior statements

Abstract nodes:
- core belief clusters
- emotion clusters
- behavior patterns
- topic families

Edge types:
- `history_to_belief`
- `belief_to_intermediate`
- `intermediate_to_thought`
- `situation_to_thought`
- `thought_to_emotion`
- `thought_to_behavior`
- `coping_to_behavior`
- `instance_of`

**Minimum viable version**
If you want to start without rebuilding everything:

1. Keep the current graph.
2. Add statement-level nodes from Patient-psi.
3. Add mappings from those statements to current abstract nodes.
4. Build:
- `extract_external_from_post.py`
- `retrieve_case_context.py`
- `construct_case.py`
- `validate_case.py`

That is enough for a first usable expansion pipeline.

**Main methodological rule**
Treat Reddit posts as:
- evidence for the external episode

Treat the graph as:
- prior structure for plausible internal formulation

Treat the generated case as:
- constrained synthesis, not direct transcription

If you want, I can next give you the exact JSON schema for the hybrid graph and the four scripts.