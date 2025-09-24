# Natural Disaster BenchMarked EarthMind

This repo expands on the GEO Bench-VLM, novelly assessing EarthMind’s performance
against other contemporary tested models below. I have discussed my primary findings below.
Semantic Segmentation: Most notably, EarthMind addresses one of the key limitations you
highlighted: it supports semantic segmentation, enabling direct evaluation on tasks formerly
untestable by GEO Bench-VLM. EarthMind’s mIoU on segmentation tasks is slightly below
GLAMM at around ~0.11. However, I also conducted some preliminary testing on specific
object classes and observed encouraging segmentation results for key categories such as houses
and roads.

- Single Downstream Tasks: On MCQ evaluation, EarthMind shows inconsistent performance,
performing particularly poorly on natural disaster scenarios, likely due to limited exposure to
datasets like xBD. I also analyzed prompt variability and compiled the statistics in this
spreadsheet, following a similar approach to your paper.
- Captioning: EarthMind achieves a BERTScore of approximately 0.87 in captioning, significantly
surpassing Sphinxs 0.6453, which highlights EarthMind’s strength in producing semantically
meaningful descriptions.

This benchmark, along with this additional analysis reveal important gaps in downstream task
performance and semantic segmentation. Further work is needed, including fine-tuning with
datasets like the recent DM3 natural disaster dataset, to improve EarthMind’s capabilities and
real-world applicability. I hope these observations from evaluating EarthMind can be helpful in
supporting ongoing work to improve geospatial VLM.
