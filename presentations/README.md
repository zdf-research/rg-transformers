# presentations/

Weekly seminar slides and derivation notes. One folder per session, named by week number and presenter.

## Structure

```
presentations/
└── week03-alice/
    ├── slides.pdf      ← exported slides (required)
    └── notes.md        ← optional: extended derivations, links, things you ran out of time for
```

## Naming convention

```
weekNN-your-name/
```

Examples: `week01-alice/`, `week07-bob/`, `week12-carol/`

Use the two-digit week number so folders sort correctly in the file browser.

## Format

**Export slides to PDF before pushing.** This ensures they render correctly for everyone regardless of what you used to make them — PowerPoint, Keynote, Google Slides, Beamer, or a napkin photographed at the right angle. A PDF that opens without the original software is infinitely more useful to the group as a reference six months later.

If you have derivations, extra references, or things you wanted to cover but ran out of time, add them in a `notes.md` alongside the slides. This is where the document becomes genuinely useful as an archive.

## Pushing

```bash
mkdir -p presentations/week03-your-name
cp your_slides.pdf presentations/week03-your-name/slides.pdf
git add presentations/week03-your-name/
git commit -m "week03: add attention presentation slides"
git push origin member/your-name
```

Presentations can be pushed directly from your personal branch — no PR needed.
