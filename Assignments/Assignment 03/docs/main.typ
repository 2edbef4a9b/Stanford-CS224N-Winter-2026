#set page(
  header: context block(
    outset: (top: 0pt, right: 0pt, bottom: 6pt, left: 0pt),
    stroke: (bottom: 0.25pt),
  )[
    #set text(size: 10pt)
    #grid(
      columns: (1fr, auto, 1fr),
      align: (left, center, right),
      [CS 224n],
      [Assignment 03],
      [Page #counter(page).display("1 of 1", both: true)],
    )
  ],
)

#set heading(numbering: "1.")
#set enum(numbering: "a.i.")
#set math.equation(numbering: "(1)")

#set text(
  font: "EB Garamond",
  size: 12pt,
  weight: "regular",
)

#show math.equation: set text(font: "Euler Math")
#show cite: set text(fill: rgb("#A68DE9"))
#show ref: set text(fill: rgb("#A68DE9"))
#show footnote: set text(fill: rgb("#A68DE9"))
#show link: set text(font: "VictorMono NF", fill: rgb("#A68DE9"), size: 10pt)
#show link: underline

#show raw.where(block: false): it => box(
  fill: luma(245),
  inset: (top: 0em, right: 0.2em, bottom: 0em, left: 0.2em),
  radius: 0.3em,
  outset: (top: 0.35em, right: 0.2em, bottom: 0.25em, left: 0.2em),
)[
  #text(
    font: "VictorMono NF",
    fill: rgb("#A68DE9"),
  )[#it.text]
]

#show heading.where(level: 1): set text(
  fill: rgb("#FFB3CC"),
  size: 16pt,
  weight: "semibold",
  style: "normal",
)

#show heading.where(level: 2): set text(
  fill: rgb("#FFB3CC"),
  size: 14pt,
  weight: "semibold",
  style: "normal",
)

#show heading.where(level: 1): set block(
  above: 1.25em,
  below: 1em,
)

#show heading.where(level: 2): set block(
  above: 1.25em,
  below: 1em,
)

#show figure.caption: it => block(width: 101%)[
  #set text(size: 10pt)
  #set align(center)
  #text(weight: "bold", fill: rgb("#A68DE9"))[
    #it.supplement
    #context it.counter.display(it.numbering).
  ] #it.body
]

#align(center)[
  #text(
    font: "Spectral",
    size: 22pt,
    weight: "bold",
  )[CS 224N Winter 2026 Assignment 03 \
    Self-Attention and Transformers]

  #text(size: 12pt, fill: rgb("#A68DE9"))[#(
    datetime.today().display("[month repr:long] [day], [year]")
  )]
]

#v(14pt)

#set par(justify: true)

This assignment is an investigation into Transformers, the prevailing architecture used for frontier LLMs.

The pset has three questions:

- In the first, you will gain intuition about how the self-attention mechanism in transformers works
- In part two, you will derive some properties of positional encodings.
- In the third part, you will code a transformer (almost) from scratch, and start training it on your laptop.

#text(weight: "bold")[
  Please tag the questions correctly on Gradescope, otherwise the TAs will take points off if you don't tag questions.
]

#v(15pt)

#text(weight: "bold")[For code submission], run `bash create_submission.sh`, which will zip your `model_solution.py`, `train.py`, and `utils.py`. Then directly upload the resultant `submission.zip` to Gradescope.

#pagebreak()

#include "sections/attention.typ"

#pagebreak()

#include "sections/postition.typ"

#pagebreak()

#include "sections/coding.typ"
