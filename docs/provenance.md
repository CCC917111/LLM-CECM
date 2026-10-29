# Provenance of the retained summaries

The framework, result CSVs and images were first published together in
[commit 7734752](https://github.com/CCC917111/LLM-CECM/commit/77347524ccd579708a785409b57a690ba6a17a49).
Its commit message records AI assistance. This identifies how the files entered the repository; it does not
establish whether each file was copied from the paper, exported from an original experiment, reconstructed,
or created during documentation work. AI assistance alone does not establish that a value is false either.

## Source checks

The published abstract, available through the [publisher record](https://www.sciencedirect.com/science/article/pii/S0960148126009651)
and [author preprint record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6643509), reports:

- 80.5% fewer LLM calls and 81.5% less runtime with adaptive decision frequency.
- A rolling price-volatility proxy decreasing from 5.49 to 2.69 CNY/MWh.

These are abstract-level source checks, not independent reproduction of the effects.
The full paper was not obtained for this review. In particular, the repository's round-31 convergence,
72-round horizon and +12.3% profit figures have not yet been matched to a specific paper page or table.
The absolute call/runtime totals, MAPE values, repeated-run statistics and individual image origins likewise
remain unverified against the full source. A percentage in an abstract does not verify the absolute totals
in a repository table. Failure to obtain the full text does not establish that a claim is absent from it.

## What to confirm with the contributors

Start with the final paper or original report, any pre-publication source-code folder, and any original
experiment outputs the contributors actually retain. Establish the source of each retained table before
requesting run records on the assumption that its stated experiment was performed exactly as described.
The longer [reproducibility checklist](reproducibility.md) describes what would be needed to reproduce or
audit a confirmed experiment; it is not evidence that those original materials exist.

The current engineering fixes and offline tests do not validate the provenance of the old result files.
