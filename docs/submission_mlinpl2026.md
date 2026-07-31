# ML in PL Conference 2026 — Submission

**Venue:** ML in PL Conference 2026, Warsaw, Poland (Oct 08 2026)  
**Submission deadline:** Aug 03 2026 11:59AM UTC  
**OpenReview:** https://conference.mlinpl.org/2026/about-ml-in-pl-conference

---

## Title

Longitudinal Multimodal Multiple-Instance Learning of Routine Biopsy Surveillance Predicts Rejection, Chronic Dysfunction and Mortality after Lung Transplantation

## Short Title

Longitudinal Multimodal MIL for Lung Transplant Outcomes

## Authors

Dinesh Adhithya Haridoss

## Abstract

Lung transplantation has the poorest long-term survival of any solid-organ transplant, driven by acute cellular rejection (ACR) and chronic lung allograft dysfunction (CLAD). We present a longitudinal multimodal multiple-instance learning (MIL) framework that integrates four routine surveillance streams — transbronchial H&E histology, bronchoalveolar lavage (BAL) cytology, thoracic CT and structured clinical variables — and explicitly models the ordered sequence of biopsy visits per patient. Benchmarking eight architectures across four clinical endpoints under strict 5-split × 4-fold nested cross-validation in ~350 recipients, longitudinal models achieved a concordance index of 0.771 ± 0.056 for mortality and 0.679 ± 0.064 for time-to-next-ACR, exceeding multivariate linear baselines by 19 and 9 concordance points respectively. A learned per-task biopsy-weighting network discovered, without temporal supervision, opposite temporal windows for related ACR endpoints — early visits for long-term rejection risk, recent visits for current-grade classification — recapitulating established transplant immunology. Attention attribution revealed a reproducible macrophage biology: tissue-resident alveolar macrophages (TRAM) mark survivors, while monocyte-derived macrophages (MoAM) and CT structural-deterioration patterns mark high mortality risk, reproduced across all five independent data splits.

## Keywords

multiple instance learning, multimodal learning, computational pathology, lung transplantation, survival analysis, set transformer, longitudinal learning, attention mechanisms

## Area of Submission

AI4Science, Computational Biology, Multimodal Models, Computer Vision

---

## Presenter Information

- **Affiliation:** Helmholtz Munich
- **Academic Degree:** PhD and higher
- **Job Position:** None / Student
- **Job Role:** Researcher
- **Field of Interest:** Deep Learning, Computer Vision, Computational Biology

**Biography:**
Dinesh Adhithya Haridoss is a researcher at Helmholtz Munich working at the intersection of computational pathology and clinical machine learning. His work focuses on multimodal deep learning for post-transplant outcome prediction, integrating histology, radiology, cytology and clinical time series under multiple-instance learning frameworks. He is currently developing longitudinal architectures that exploit the full structured surveillance record generated during routine transplant follow-up to predict rejection, chronic allograft dysfunction and mortality.

## Call for Contributions

- **Call for Talks:** Yes, only for Main Conference
- **Call for Posters:** Only if submission for Call for Talks is rejected
