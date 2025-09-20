# Large Language Model Use

We used large language models (Gemini 2.5 Flash and Pro) to help polish the writing and to find some related works.
Github Copilot's code-suggestions feature was used during writing of the source code.
We reviewed all suggested changes and inclusions for factual accuracy, clarity, conciseness, and style.
We cannot provide prompts for information retrieval and discovery because we conducted this on an ad hoc basis.

The prompt used to help polish the writing is as follows:

```markdown
Act as an academic editor for the following text. We will provide you with a text, and you will edit it according to the following instructions:
* Improve grammar.
* Keep latex markup intact.
* Use the pronoun ``we'' to refer to this text's authors.
* Avoid passive voice.
* Make the text more concise.
* Remove weasel words.
* Remove lexical illusions.
* Point out factual errors.
* Keep abbreviations.
* Remove abuse of the passive voice.
* Opening quotes should be ``.
* Remove lexical illusions.
* Use short words whenever appropriate.
* When referring to the ideas of others, use present tense.
* Prefer 'is' over 'comprises'.
* Grammatically treat equations as sentences.
* Use New Zealand English.
* Do NOT use markdown.
* Avoid starting multiple sentences with the same word.
* Use `---` to indicate an em dash.
* Use `--` to indicate an en dash.
* Use `-` to indicate a hyphen.
* Avoid semi-colons
```