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
      [Assignment 02],
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
  )[CS 224N Winter 2026 Assignment 02 \
    Word2Vec & Dependency Parsing]

  #text(size: 12pt, fill: rgb("#A68DE9"))[#(
    datetime.today().display("[month repr:long] [day], [year]")
  )]
]

#v(14pt)

#set par(justify: true)

In this assignment, you will review the mathematics behind Word3Vec and build a neural dependency parser using PyTorch. For a review of the fundamentals of PyTorch, please check out the PyTorch review session on Canvas. In Part 1, you will explore the partial derivatives involved in training a Word2vec model using the naive softmax loss. In Part 2, you will learn about two general neural network techniques (Adam Optimization and Dropout). In Part 3, you will implement and train a dependency parser using the techniques from Part 2, before analyzing a few erroneous dependency parses.

#include "sections/word2vec.typ"

#pagebreak()

#include "sections/optimization.typ"

#pagebreak()

#include "sections/parsing.typ"
