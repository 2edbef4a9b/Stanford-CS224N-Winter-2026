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
      [Assignment 04],
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
#show cite: set text(fill: rgb("#C58DE9"))
#show ref: set text(fill: rgb("#C58DE9"))
#show footnote: set text(fill: rgb("#C58DE9"))
#show link: set text(font: "VictorMono NF", fill: rgb("#C58DE9"), size: 10pt)
#show link: underline

#show raw.where(block: false): box.with(
  fill: luma(250),
  inset: (top: 0em, right: 0.25em, bottom: 0em, left: 0.25em),
  radius: 0.3em,
  outset: (top: 0.35em, right: 0.0em, bottom: 0.25em, left: 0.0em),
)

#show raw.where(block: false): set text(
  fill: rgb("#FF8863"),
)

#show raw.where(block: true): block.with(
  fill: luma(250),
  inset: 1em,
  radius: 0.5em,
  width: 100%,
)

#show raw: set text(
  font: "VictorMono NF",
)

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
  #text(weight: "bold", fill: rgb("#C58DE9"))[
    #it.supplement
    #context it.counter.display(it.numbering).
  ] #it.body
]

#align(center)[
  #text(
    font: "Spectral",
    size: 22pt,
    weight: "bold",
  )[CS 224N Winter 2026 Assignment 04 \
    LLM Evals]

  #text(size: 12pt, fill: rgb("#C58DE9"))[#(
    datetime.today().display("[month repr:long] [day], [year]")
  )]
]

#v(14pt)

#set par(justify: true)

In this assignment, you will explore different methods to evaluate the performance of LLMs. We can roughly split LLM evaluations into four categories:

- *Standard closed ended benchmarking.* In this type of benchmarking, we produce a set of problems and an associated, usually simple, way of classifying whether the answer to a problem is correct. For example, we may produce a set of math problems and evaluate a response as correct if it ends with a string that matches the correct numerical answer. Examples of such benchmarks include GSM8K @cobbe2021training and MATH @hendrycks2021measuring.
- *LLM as judge.* In some cases, it is difficult to create a simple verifier to classify whether an answer has some property. Sometimes, we can use a separate, possibly more powerful, LLM to decide. For example, @souly2024strongreject use GPT-4o to categorize model outputs as “harmful and helpful,” a metric used to evaluate the susceptibility of LLMs to jailbreaking.
- *User study.* The model is evaluated by a collection of humans in a controlled experiment. These types of evaluations are often expensive and difficult to run, but if conducted properly they provide very strong signal. For example, @becker2025measuring conduct a user study investigating the effect on coding productivity of AI assistants using a small set of open-source software engineers.
- *Interacting with the model.* This is the informal testing we all do when we actually use LLMs. There is no particular structure to it, instead being open-ended and exploratory. Simply interacting with a model can be the fastest way to find different failure modes.

This problem set involves three different parts:

- In the first part, you will get access to a number of different LLMs through an API. Your job is to run some standard closed ended benchmarking. There is also the opportunity for extra credit here.
- In the second part, you will conduct some LLM-as-judge benchmarking.
- In the third part, you will red-team an LLM to try to show it can be made to disobey its system prompt.

#pagebreak()

#include "sections/claiming.typ"

#pagebreak()

#include "sections/standard.typ"

#pagebreak()

#include "sections/judge.typ"

#pagebreak()

#include "sections/interaction.typ"

#pagebreak()

= Submission Instructions

- Please submit the written component as a PDF with tagged pages on Gradescope. *Please tag the questions correctly.*
- For code submission, please run `bash create_submission.sh` and upload your code files submission.

#bibliography(
  "assets/main.bib",
  style: "apa",
  title: [References],
)
