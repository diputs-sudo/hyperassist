# HyperAssist Formula Library

Welcome to the **formula vault** for HyperAssist!  
If you’re reading this, you’re probably tired of random “magic numbers” and cryptic hyperparameter folklore.  
This folder is here so you (and I) never have to guess *why* a particular value or trick is used in deep learning training.

---

## What’s in Here?

- **L1:** The basics. Classic rules, foundational formulas, and stuff everyone should know (but few really do).
- **L2:** Smart heuristics and scaling tricks, modern best practices that work across most projects.
- **L3:** Transformers, attention, dropout, smoothing, and the recipes powering today’s big models.
- **L4:** Where the theory gets real. PAC-Bayes, curvature-aware, Fisher/NTK, information bottleneck stuff you won’t see in most codebases.

Every file:
- Starts with a formula (real math)
- Has an intuitive, plain English explanation
- Gives you a code snippet to plug in right away
- Shares tips, warnings, and “why this works”

**Want to see real working code for each formula and level?  
Check out:**  
`hyperassist/hyperassist/parameter_assist/formulas/` folder! 
(You’ll find Python implementations for every formula.)

---

## Why Does This Exist?

Because most guides just say  
“Try batch size 32, learning rate 1e-3, and good luck!”

But I wanted something better.  
Here, every formula is:
- **Grounded in theory or evidence**  
- **Usable in real projects**  
- **Explained for both practitioners and nerds**  
- **Not just “do X”—but “here’s why”**

If you’re new: start at L1 and work up.  
If you’re a researcher: skip to L4 and dig in!

---

## Who’s This For?

- Anyone who trains deep learning models and wants more than “copy paste and pray.”
- Engineers who want *good defaults* with actual reasons.
- Curious researchers looking for theory they can actually use.

---

## What if I Disagree?

Perfect! File an issue, PR, or just email me (diputs-sudo@proton.me).  
This is a living library. Better formulas, evidence, or counterexamples are *always* welcome.

---

## Final Words

## The Cheat Sheet Spirit

This library is the result of deep dives down the machine learning rabbit hole, so you don’t have to.  
My goal: raise the bar for AI projects everywhere, and make the real theory accessible for anyone who wants to build better models.

If this cheat sheet saves you time, teaches you something new, or helps you train smarter. That’s a win.

Enjoy, and keep raising the bar!

— diputs-sudo