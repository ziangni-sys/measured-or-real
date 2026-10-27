# Prompt Texts

## In-Pool Instructions

1. `Answer the following multiple-choice question. Respond with only the letter of the correct option.`
2. `Choose the correct option. Reply with a single letter (A, B, C, or D).`
3. `Read the question and pick the best answer. Output only the option letter.`
4. `Which option is correct? Answer with the letter only.`

## Option Orders

The four zero-based permutations are `(0,1,2,3)`, `(3,2,1,0)`,
`(1,2,3,0)`, and `(2,3,0,1)`. The shot counts are zero and two.

## Two Demonstrations

```text
Question: Which gas do plants absorb from the air?
A. Oxygen
B. Carbon dioxide
C. Nitrogen
D. Helium
Answer: B

Question: What is the boiling point of water at sea level?
A. 50C
B. 90C
C. 100C
D. 150C
Answer: C
```

The inline condition places this block before the target question in one user
message. The chat-format control presents each demonstration question as a user
turn and its answer letter as the following assistant turn.

## Four Out-of-Pool Instructions

1. `You will be given a question with four answer options. Decide which one is correct and reply with its letter alone.`
2. `Below is a quiz item. State the letter of the right answer and nothing more.`
3. `Consider the question and the options, then give exactly one letter (A, B, C, or D) as your answer.`
4. `Identify the correct choice. Your entire response must be a single option letter.`

The source-of-truth constants are also present in `scripts/llm_decomp.py`,
`scripts/shot_format.py`, and `scripts/holdout_wordings.py`.
