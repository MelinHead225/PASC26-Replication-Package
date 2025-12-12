# PASC26-Replication-Package

This repository is an artifact from the research paper:

Multi-Artifact Analysis of Self-Admitted Technical Debt in
Scientific Software.

Context: Self-admitted technical debt (SATD) occurs when develop-
ers explicitly acknowledge shortcuts or compromises in software, of-
ten documented in code comments, commits, pull requests, or issue
tracking systems. While SATD has been studied in general-purpose
software, scientific software (SSW) introduces unique challenges, as
compromises can directly affect the validity and reproducibility of
scientific results. Objective: This study aims to identify, categorize,
and evaluate scientific debt, a specialized form of SATD in SSW, and
to assess the extent to which traditional SATD categories capture
these domain-specific issues. Method: We perform a multi-artifact
analysis across code comments, commit messages, pull requests,
and issue tracking sections from 23 open-source SSW projects. We
construct and validate a curated dataset of scientific debt, perform
SATD classification experiments, and conduct a practitioner valida-
tion to assess the practical relevance of scientific debt. Results: Our
multi-source SATD classifier achieves strong overall performance
(accuracy = 0.916, macro F1 = 0.826). Across 900,358 artifacts from
23 CASS projects, pull requests (12.0%) and issue trackers (9.6%)
show the highest SATD prevalence, demonstrating the importance
for multi-artifact analysis. Models trained only on traditional SATD
types misclassify most scientific debt instances (74.4%) as non-debt,
highlighting the need to explicitly identify this domain-specific debt.
A study validation of seven DOE scientists and researchers showed
that participants agreed with our scientific debt annotations and
rated the label’s practical usefulness at 4.03/5, demonstrating that
scientific debt is both recognizable and actionable in real workflows.
Conclusions: Scientific debt represents a unique form of SATD in
SSW that requires specialized identification and management. Our
dataset, classification analysis, and practioner validation results
provide the first formal multi-artifact perspective on scientific debt,
highlighting the need for tailored SATD detection approaches in
SSW.

For any questions or inquires please reach out to ericmelin@u.boisestate.edu.
