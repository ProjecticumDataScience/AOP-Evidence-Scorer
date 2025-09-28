
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Adverse Outcome Pathways  by WoE</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>

  <header>
    <h1>AOP-main</h1>
    <p>
      This repository focuses on <strong>entity extraction</strong> and
      <strong>biological plausibility evaluation</strong> from scientific articles
      related to Adverse Outcome Pathways (AOP). It processes curated data about
      Key Event Relationships (KERs), extracts relevant biological entities, and
      evaluates mechanistic plausibility.
    </p>
  </header>

  <nav aria-label="Table of contents">
    <h2>Table of Contents</h2>
    <ol>
      <li><a href="#overview">Overview</a></li>
      <li><a href="#repository-structure">Repository Structure</a></li>
      <li><a href="#input-data">Input Data</a></li>
      <li><a href="#folders">Folders</a></li>
      <li><a href="#workflow">Workflow</a></li>
      <li><a href="#future-work">Future Work</a></li>
    </ol>
  </nav>

  <main>
    <section id="overview">
      <h2>Overview</h2>
      <p>
        The project enables the extraction of key biological entities (e.g., stressor,
        species, evidence levels, correlations) and the evaluation of biological
        plausibility between events. It leverages manually curated ground-truth data to
        validate the performance of an automated pipeline and stores results in a
        structured format.
      </p>
    </section>

    <section id="repository-structure">
      <h2>Repository Structure</h2>

      <pre><code>AOP-main/
│
├── README.md                  # Project documentation (this file)
│
├── data_input/                # Input data for entity extraction & evaluation
│   └── Data_evaluatie_accuraatheid_agent_Daan_de_Jong_16-06-2025.xlsx
│
├── data_output/               # Processed outputs and extracted results
│   ├── entities_ner.csv
│   └── entities_ner.parquet
│
└── src/                       # Source code for entity extraction pipeline
    └── extract_entities_ner.py
</code></pre>
    </section>

    <section id="input-data">
      <h2>Input Data</h2>

      <h3 id="data-file">Data_evaluatie_accuraatheid_agent_Daan_de_Jong_16-06-2025.xlsx</h3>
      <p>
        This Excel file is the <strong>core dataset</strong> of the project. It contains
        both manually curated and automatically generated information used for evaluation.
        It also <strong>contains the articles (and structured article-derived records) on
        which all subsequent work is based</strong> — namely, <em>entity extraction</em>
        and <em>assessment of biological plausibility</em>.
      </p>

      <p><strong>Sheets and purpose:</strong></p>
      <ul>
        <li>
          <strong>GroundTruth kers</strong> — Manually curated KERs with PMIDs, stressors,
          species, evidence levels, and correlation types (gold standard).
        </li>
        <li>
          <strong>GroundTruth s4</strong> — Aggregated KER-level statistics (counts of
          evidence, unique stressors/species/systems) and a normalized biological
          plausibility score.
        </li>
        <li>
          <strong>GroundTruth kers filtered</strong> / <strong>GroundTruth s4 filtered</strong>
          — High-confidence subsets of the above.
        </li>
        <li>
          <strong>Agent eval results</strong> — Outputs from the automated evaluation
          pipeline: extracted entities/relations, biological plausibility scores, paper
          snippets, and metadata (e.g., model used).
        </li>
      </ul>

      <p>
        In short, this file acts as both the <strong>input (ground truth and article-driven data)</strong>
        and the <strong>benchmark</strong> for evaluating the pipeline’s accuracy.
      </p>
    </section>

    <section id="folders">
      <h2>Folders</h2>

      <article id="folder-data-input">
        <h3><code>data_input/</code></h3>
        <p>
          Holds all raw input data for the pipeline. Currently includes the Excel file
          described above, which provides the articles and curated KERs used for
          downstream entity extraction and plausibility evaluation.
        </p>
      </article>

      <article id="folder-data-output">
        <h3><code>data_output/</code></h3>
        <p>Contains processed results after running the extraction pipeline:</p>
        <ul>
          <li><strong>entities_ner.csv</strong> — Extracted entities in CSV format for rapid inspection.</li>
          <li><strong>entities_ner.parquet</strong> — The same results stored in Parquet for efficient IO and analysis.</li>
        </ul>
      </article>

      <article id="folder-src">
        <h3><code>src/</code></h3>
        <p>Source code for the entity extraction pipeline:</p>
        <ul>
          <li>
            <strong>extract_entities_ner.py</strong> — Implements the Named Entity Recognition (NER) flow
            that processes article text to extract relevant biological entities (species, stressors, key events),
            writing results into <code>data_output/</code>.
          </li>
        </ul>
      </article>
    </section>

    <section id="workflow">
      <h2>Workflow</h2>
      <ol>
        <li>
          <strong>Prepare Input</strong> — Load the Excel file containing curated KERs, article references,
          and evaluation scaffolding.
        </li>
        <li>
          <strong>Run Entity Extraction</strong> — Execute <code>src/extract_entities_ner.py</code> to process
          article text and extract entities.
        </li>
        <li>
          <strong>Evaluate</strong> — Compare extracted entities and plausibility scores against the ground truth
          data to assess accuracy and consistency.
        </li>
        <li>
          <strong>Store Results</strong> — Save outputs to CSV and Parquet in <code>data_output/</code> for
          analysis and reporting.
        </li>
      </ol>
    </section>

    <section id="future-work">
      <h2>Future Work</h2>
      <ul>
        <li>Expand dataset coverage with additional KERs and supporting papers.</li>
        <li>Improve NER model accuracy and coverage of biological entities.</li>
        <li>Add a simple CLI for running the pipeline end-to-end.</li>
        <li>Automate plausibility scoring and reporting artifacts.</li>
      </ul>
    </section>
  </main>

</body>
</html>
