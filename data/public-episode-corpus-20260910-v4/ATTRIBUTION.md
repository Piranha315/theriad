# Source attribution and adaptations

- HumanEval: OpenAI, MIT; https://github.com/openai/human-eval . Original prompts/tests preserved; complete-code instruction added.
- GSM8K: OpenAI, MIT; https://github.com/openai/grade-school-math . Original questions preserved; final-number instruction added.
- HotpotQA: Yang et al. (EMNLP 2018), CC BY-SA 4.0; https://hotpotqa.github.io/ . Mirror: https://huggingface.co/datasets/hotpotqa/hotpot_qa . Derived HotpotQA rows/passages remain CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ . Four bounded source paragraphs are retained and strict-answer instructions added. Wikipedia article links occur in each fixture; these are historical corpus snapshots, not today's pages. No endorsement is implied.
- tau2-bench: Sierra Research (2025), MIT; https://github.com/sierra-research/tau2-bench . Native task/policy files retained unchanged for adapter development; no independence count assigned.

Exact revisions, retrieval URLs, SHA-256 values, original READMEs and MIT license files are in sources/ and source-manifest.json. The initial CMU HotpotQA endpoint timed out; its pinned public HF mirror was used. No blanket repository license overrides these source licenses.
