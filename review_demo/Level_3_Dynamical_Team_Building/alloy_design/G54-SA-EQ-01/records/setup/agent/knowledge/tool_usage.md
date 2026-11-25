# Calling the scientific functions and reading their records

Three registered functions are available: evaluate_coarsening(compositions),
evaluate_phase_stability(compositions), evaluate_solidification(compositions).
Each accepts a list of dictionaries with Ni, Er and Zr atomic percentages;
Al is the balance. An empty list requests only the fixed reference. The reference
uses its original mass percentages internally. Use at most three candidates per
batch for this comparison and respect the total allowances in the task.

The three functions are also available from the ordinary Python executor:

```python
from science_api import tools
# compositions is your chosen list of candidate dictionaries.
result = tools.evaluate_coarsening(compositions)
print(result)
```

The other methods have the same argument. Each response contains reference,
results, status, usage, method_fingerprint and links to complete records.
Read records using ordinary Python; no separate evidence-reading tool is required:

```python
import json
from science_api import REPO
# relative_record is a path returned by a scientific function.
record = json.loads((REPO / relative_record).read_text())
```

Use returned record paths as evidence identifiers in your comparison. They refer
to your calculations, not to prior designs. Do not look outside the supplied
references or this investigation's records. Python variables do not survive
between executor scripts; save useful tables to files and reload them as needed.

Repeated calculations in the same output directory reuse matching cached results.
Cache hits do not constitute fresh verification. The existing scientific class
supports fresh calculations by using a new output directory. If you decide to
perform the task's final recalculation, use the same three methods there:

```python
from science_api import AlloyDesignTools, OUTPUT
verification = AlloyDesignTools(OUTPUT / "fresh_verification")
# Supply your selected composition, not an example answer.
coarsening = verification.evaluate_coarsening(compositions)
phase = verification.evaluate_phase_stability(compositions)
solidification = verification.evaluate_solidification(compositions)
print(coarsening)
print(phase)
print(solidification)
```

Each includes the reference. Compare actual returned values and statuses to the
earlier records, then assess the scientific interpretation. Reusing that directory
again will reuse its cache. No recalculation is automatically initiated for you.
