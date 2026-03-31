#let mtext = text.with(font: "Spectral", size: 12pt, weight: "regular")

#let unjustified(body) = {
  set par(justify: false)
  body
}

#let proof = text(
  fill: rgb("#C58DE9"),
  size: 12pt,
  font: "Space Grotesk",
  weight: "semibold",
)[Proof:]

#let answer = text(
  fill: rgb("#C58DE9"),
  size: 12pt,
  font: "Space Grotesk",
  weight: "semibold",
)[Answer:]
