# After the First Touch

*What Decides the Second Ball at Set Pieces*: research for the **Hudl Performance Insights 2026**
call.

- Call: <https://www.hudl.com/blog/hpi-2026-research-competition>
- Finalists present on **13 November 2026**, Emirates Stadium, London.

## The question

Every set-piece coach plans the delivery. Far fewer plan what happens two seconds later, when
the first touch does not settle the play: a defender heads the cross clear, a teammate flicks it
on, a shot is blocked, the keeper parries. The ball drops.

The received wisdom is that what follows is noise, and that is why both the routines and the
published models stop at the first contact. This work asks whether that noise has a trainable
shape, and whether the chaos can be generated on purpose rather than waited for.

## What it finds

Over the 2025/26 season of the five big European leagues, about two in five corner goals come after
the first touch, and per shot the value sits in the attacking touch that keeps the ball alive (a
flick-on or a prolongation), not in a ball won back after a clearance. Where the players stand when
the corner is taken predicts where the ball is aimed, but once the delivery is known it adds
nothing measurable to how the corner ends, while the same method finds a clear signal in who wins
the first contact. The positions at the first contact do add to the delivery in telling who wins the
second ball, too late to be planned. The figures, with their intervals, are in the artefacts the
pipeline writes.

## What is here

The code that goes from the granted export to every table and figure of the paper, and the tests
that guard it. `pipeline/run_e2e.py` runs every stage in order and regenerates every artefact
(`--list` prints the stages), given the granted export, which is not distributed.

## Running it without the data

The granted export is not in this repository, and neither are the private research notes of the
`docs/` base. A clone reads the code and runs the tests that need neither: a test marked
`requires_data` or `requires_sb` skips when the export is missing, and a test that reads a private
note or the generated record skips with its reason. Python 3.10 or later; the dependencies are
listed in `pyproject.toml`, with `torch` (the two neural drivers) under the optional group `gpu` and
`python-docx` and `lxml` (the paper build) under `paper`; without them the tests of those modules
are skipped with their reason.

```bash
python -m pytest tests/ -q
```

## Layout

```
src/          the stages, each one a module with a single responsibility
pipeline/     the drivers that run them end to end
tests/        the modules' and the drivers' tests
renders/      figure generation, never committed output
docs/         architecture, data schema and definitions; the rest of the base is private
refdata/      small versioned reference tables
figures/      fonts, icons and the author's logo; generated images are gitignored
LICENSE       MIT
```

## Conventions

- Documentation in English.
- `data/` is gitignored: the granted export is never committed.
- `docs/` holds the architecture, the data schema and the definitions. The research notes of the
  author's working copy are private and not published, so a comment citing one (`evidence.md`,
  `measured.md`, or the provider reference under `docs/hudl/`) points at a source this repository
  does not carry.
- Numbers live in the artefacts the pipeline writes, not copied into prose where they go stale.

## License

MIT, see `LICENSE`.
