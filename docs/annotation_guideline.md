# Annotation guideline for Azerbaijani sentiment labeling

**Edition:** current edition, November 2026.

This guideline is a working document that the group maintains across its
annotation projects; earlier editions were not archived. It is released here in
its current edition, translated into English from the Turkish working text. It
states the rules in their present form. Section III-C of the article documents
how the two annotation batches of the benchmark were carried out.

The Azerbaijani examples in this document were written for training purposes.
They are not records from the dataset.

## 1. Task and scope

Read the whole comment and label the overall evaluation that the writer
expresses as `NEG`, `NEU` or `POS`. The unit is a single comment; sentences and
product aspects are not labeled separately. The same rules apply in the
technology, finance, social content, public services and retail domains.

Rely only on the text shown to you. Do not assume the parent post, which you
cannot see, the writer's intention, or your own knowledge of a brand. Checking
whether a claim is true is not part of the task. Do not change the spelling,
emoji or punctuation of a comment.

## 2. Labels

| Code | Meaning | When to choose it |
|---|---|---|
| `NEG` | Negative | Dissatisfaction, criticism, rejection, disappointment or a clearly negative experience dominates. |
| `NEU` | Neutral | Information without an evaluation, or a genuine request for information. The coding rule below applies to balanced mixed comments. |
| `POS` | Positive | Satisfaction, liking, praise, appreciation, support or a positive recommendation dominates. |

**Mixed sentiment.** When positive and negative evaluations occur together,
follow the explicit overall verdict first and the dominant direction second. If
the two are truly balanced, choose `NEU` and write "mixed" in the note field.
This guideline adopts that coding rule for balanced mixed comments; mixed
content and neutral content without sentiment are told apart through the note.

**Uncertainty.** Failing to resolve the meaning or the direction of sentiment is
not by itself a reason for `NEU`. Set the review flag and describe the problem
briefly. "Review" is a workflow state, not a fourth sentiment class.

## 3. Short procedure for every comment

1. Read the whole text; check negation, contrast, quotation and emoji.
2. Identify what the writer evaluates and the writer's overall stance. Do not
   decide by counting words.
3. Choose the one label you can defend; if you cannot, set the review flag. The
   reason for your label must be visible in the text.
4. Add the note "mixed" for mixed cases, and the review flag with a short reason
   for unresolved ones.

## 4. Decision rules for borderline cases

### Negation suffixes and words

Look at which action or evaluation the negation affects. *Razı deyiləm* ("I am
not satisfied") expresses dissatisfaction. *Pis deyil, çox bəyəndim* ("Not bad,
I liked it very much") is a positive evaluation. *Tədbirə qatılmadım* ("I did
not attend the event") only reports non-attendance. The presence of *-ma/-mə*,
*-mır/-mir*, *deyil* or *yox* does not by itself decide `NEG`.

### Statements of fact and implicit complaints

Stating a fact is not always neutral. *Komissiya 2 manatdır* ("The commission is
2 manats") is fee information. *Xəbərim olmadan hesabımdan pul çıxılıb* ("Money
was taken from my account without my knowledge") objects to an unwanted event.
`NEG` may be used when a fault, an unmet service or a clear grievance can be
understood from the text. Do not infer dissatisfaction from a fee, a duration or
a number alone.

### Questions and politeness

A genuine request for information is `NEU` when there is no other evaluation. A
question that contains a complaint can be `NEG`. *Zəhmət olmasa* ("please"), a
greeting or a routine thank-you does not make the whole comment positive;
explicit appreciation and satisfaction indicate `POS`. A question or a thank-you
added at the end of a complaint does not cancel the complaint.

### Mixed comments and several targets

Give priority to the writer's explicit overall verdict, such as *ümumilikdə
razıyam* ("overall I am satisfied") or *məsləhət görmürəm* ("I do not recommend
it"). Without such a verdict, weigh the strength of the expressions and their
share of the comment. Do not give automatic priority to the last sentence or to
what follows *amma* ("but"). If no direction dominates, use `NEU` with the note
"mixed"; if you are not sure about the balance, send the comment to review.

### Irony and mockery

An ironic reading must be supported by an explicit contrast inside the text.
*Əla xidmət! İki saat gözlədim, işim həll olunmadı* ("Great service! I waited
two hours and my issue was not resolved") is negative. Do not treat *Əla!*
("Great!") on its own as mockery by assuming outside context. If both a sincere
and a mocking reading are plausible and the text does not separate them, set the
review flag.

### Quotation and other people's opinions

If the writer states their own stance, label that stance. Reporting someone
else's opinion as news does not mean that the writer shares it. If the report is
clear and the writer adds no evaluation, use `NEU`. If it is unclear whose words
are quoted or what the writer's stance is, send the comment to review; do not
decide from quotation marks alone.

### Emoji, spelling errors and language mixing

Read emoji, exclamation marks, capital letters and lengthened letters together
with the text. They can strengthen emphasis; they do not determine a label on
their own. Intelligible Turkish, Russian or English words inside an Azerbaijani
comment are no obstacle to labeling. Do not guess for a language you cannot
resolve, an incomplete text or a reaction that depends on context.

**When sending a comment to review**, write a short reason such as "irony",
"quotation", "missing context", "unclear expression" or "no dominant direction".
If you cannot choose a defensible direction, leave the provisional label empty.
An empty record is not accepted as a final label.

## 5. Worked examples

Decisions rest only on the text shown.

| No. | Comment | Decision | Reason |
|---|---|---|---|
| 1 | *Telefonun yaddaşı 128 GB-dır.* ("The phone's storage is 128 GB.") | `NEU` | Technical specification; no evaluation. |
| 2 | *Yeniləmədən sonra tətbiq açılmır, işimi görə bilmirəm.* ("After the update the app does not open; I cannot do my work.") | `NEG` | A fault, and the user cannot do their work. |
| 3 | *Kartın illik xidmət haqqı nə qədərdir?* ("How much is the card's annual service fee?") | `NEU` | A question for information without a complaint. |
| 4 | *Niyə hər ay xəbərim olmadan hesabımdan pul çıxılır?* ("Why is money taken from my account every month without my knowledge?") | `NEG` | Objection to an unwanted deduction. |
| 5 | *Məhsuldan razı deyiləm.* ("I am not satisfied with the product.") | `NEG` | Satisfaction is negated by *deyiləm*. |
| 6 | *Bu telefon pis deyil, gündəlik istifadə üçün tam uyğundur.* ("This phone is not bad; it is fully suitable for daily use.") | `POS` | A clear positive judgment of suitability. |
| 7 | *Tədbirə qatılmadım.* ("I did not attend the event.") | `NEU` | Reports non-attendance; no sentiment stated. |
| 8 | *Çatdırılma gecikdi, amma məhsuldan çox razıyam və yenə sifariş verəcəyəm.* ("The delivery was late, but I am very satisfied with the product and will order again.") | `POS` | Overall satisfaction and the intention to buy again dominate. |
| 9 | *Dizaynı gözəldir, amma keyfiyyəti zəifdir; ümumilikdə məsləhət görmürəm.* ("The design is nice, but the quality is poor; overall I do not recommend it.") | `NEG` | The overall verdict is not to recommend. |
| 10 | *Ekranı yaxşıdır, batareyası zəifdir.* ("The screen is good, the battery is weak.") | `NEU`, note "mixed" | No dominant overall direction; mixed coding rule. |
| 11 | *Əla xidmət! İki saat gözlədim, yenə də işim həll olunmadı.* ("Great service! I waited two hours and my issue still was not resolved.") | `NEG` | The failed service in the text explains the mocking praise. |
| 12 | *Bu da belə...* ("So that is how it is...") | Review | The direction cannot be determined reliably without context. |
| 13 | *“Tədbir darıxdırıcı idi” deyirlər, amma mən çox bəyəndim.* ("They say 'the event was boring', but I liked it very much.") | `POS` | The writer's own positive evaluation is explicit. |
| 14 | *Sifarişim yenə səhv gəldi :(* ("My order arrived wrong again :(") | `NEG` | Complaint about a wrong delivery; the emoticon supports it. |
| 15 | *Support problemi tez həll etdi, çox razı qaldım.* ("Support solved the problem quickly; I was very satisfied.") | `POS` | Satisfaction is clear despite the language mixing. |

The note "mixed" is kept for example 10. For example 12 the final decision is
taken in the adjudicator's review; "Review" is not added to the target classes
of a model.

## 6. Procedure and final decision

**Start.** Select a short pilot set from the five domains. Label these examples,
which lie outside the final test set, independently first, then discuss the
neutral and mixed decisions in particular. Clarify the rules where needed and
fix the edition of the guideline. This is the recommended preparation step for
new annotation work.

**Blind annotation.** A1, A2 and A3, native speakers of Azerbaijani, work
separately. Earlier human labels, model predictions and the decisions of the
other annotators are not shown. Do not share decisions with one another during
annotation, and do not produce labels with an external source or a language
model.

**Verification.** Work in which an earlier label is visible is recorded as a
separate verification group. That label can be corrected; there is no obligation
to follow it. This process is not reported as blind re-annotation [1].

**Adjudicator's review.** A4 reviews every comment of the triple-annotated
group, including those on which the three annotators agree, and sets the final
label. The majority decision is supporting information. In a mixed case A4
applies the rule of Section 2, and records a short reason when departing from
the majority or in difficult decisions. The original labels of A1–A3 are
kept [1].

**Unresolved records.** A comment is not removed only because the annotators
disagree. If no final decision can be reached because the text is incomplete or
unintelligible, the record is held with its reason. If the data steward decides
to exclude it, the reason and the count are recorded; the record is not forced
to `NEU`.

## 7. Records and quality control

The annotator enters only the sentiment label and, where needed, the review flag
and a short note, and does not change the comment identifier or the text. For
new records the data steward also keeps the annotator identifier, the group
identifier, the type of work (blind, verification or single annotation), and the
edition and date of the guideline. The final label and A4's reason are stored in
separate fields.

Agreement is computed from the labels of A1–A3 before A4's decision; A4 is not
included. Fleiss' κ and the rate of three-way agreement are computed on the
records that carry all three labels, and pairwise Cohen's κ on the records that
carry the two labels concerned. The blind and verification groups are kept
apart. Each computation reports the number of records used and the number left
out because of a missing label [1].

Disagreements are kept on record; original labels are not overwritten to raise
agreement. A single high agreement coefficient does not show that every label is
correct. When a rule changes, the edition number is increased and the affected
examples are reviewed again [1].

**Text display.** The annotator is shown the comment with its emoji and
punctuation. The key produced for duplicate checking, from which these marks are
removed, is not used in place of the text to be labeled [1].

## Reference

[1] B. Plank, "The 'problem' of human label variation: On ground truth in data,
modeling and evaluation," in *Proc. EMNLP*, 2022, pp. 10671–10682.
https://doi.org/10.18653/v1/2022.emnlp-main.731
