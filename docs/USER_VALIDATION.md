# Shortlist directional user-validation protocol

Status: preregistered protocol; no participants or results yet.

The offline evaluation shows retrieval behavior on a specific active-user
cohort. It does not show that people prefer the recommendations. This study is
the next evidence gate. Do not change the measures or thresholds after looking
at participant results; record any protocol change as a new version first.

## Question

When a new visitor supplies five favorite movies, do they prefer Shortlist's
recommendations to a popularity list generated from the same eligible catalog?

This is a small directional product study, not a statistically powered claim
about all movie watchers.

## Participants

- Recruit 10 adults who watch movies at least monthly.
- Do not recruit project contributors or explain which list is the model output.
- Record only a participant code, session date, and the ratings below. Do not
  collect names, accounts, demographic profiles, or viewing histories.
- Replace a participant only for a predeclared procedural failure, never because
  their ratings are unfavorable.

## Locked comparison

Before the first session, record:

| Field | Locked value |
| --- | --- |
| Git commit | |
| Serving run ID | |
| Serving-bundle checksum | |
| Popularity-baseline checksum | |
| Eligible catalog size | |
| Session facilitator | |

For every participant:

1. Collect exactly five favorite movies through the normal product UI.
2. Generate 10 Shortlist results and 10 pre-cutoff popularity results from the
   same catalog. Remove the five seeds and any duplicate title across the two
   lists before presentation; replace removed entries with the next result from
   that method.
3. Randomize which method is labeled `A` or `B` per participant. Preserve the
   ranking within each list. Do not show scores, explanations, or method names
   during rating.
4. Ask the participant to rate every title independently before choosing a list.
5. Reveal neither method until the session and notes are complete.

If the current tooling cannot generate a like-for-like popularity list, stop and
add that reproducible export. Do not substitute a streaming-service trending
page or a hand-picked list.

## Measures

For each title, record:

- `would_watch`: yes/no after reading the normal movie details;
- `already_seen`: yes/no;
- `fit`: 1 (unrelated) through 5 (strongly connected to my taste);
- `novelty`: 1 (obvious/familiar) through 5 (new discovery).

After both lists, record:

- preferred list: A, B, or no preference;
- up to three titles they would save;
- one short, verbatim reason for their preference;
- any title that made the system feel broken or unsafe to trust.

Do not treat time-on-page as engagement evidence in this moderated setup.

## Preregistered analysis

Use participants, not individual titles, as the unit of comparison.

- Primary: count participants preferring Shortlist versus popularity, excluding
  `no preference` only from the preference denominator and reporting its count.
- Co-primary diagnostic: participant-level difference in `would_watch` rate,
  summarized with the median and every participant's paired difference.
- Secondary: paired differences in median fit, novelty, save count, and
  already-seen rate.
- Report all 10 participant rows. Do not report only the aggregate.
- Do not run subgroup analyses with this sample or convert the result into a
  significance claim.

Directional continuation gate: at least 7 of 10 participants prefer Shortlist,
and the median participant has a positive `would_watch` difference. Missing
either condition means inspect failure cases before expanding the study. Passing
means the concept merits a larger validation; it does not prove market value.

## Session log

| participant | A/B mapping sealed | protocol completed | preference | Shortlist would-watch | popularity would-watch | notes file |
| --- | --- | --- | --- | ---: | ---: | --- |
| P01 | | | | | | |
| P02 | | | | | | |
| P03 | | | | | | |
| P04 | | | | | | |
| P05 | | | | | | |
| P06 | | | | | | |
| P07 | | | | | | |
| P08 | | | | | | |
| P09 | | | | | | |
| P10 | | | | | | |

## Publication rule

Publish the locked fields, full participant-level table, protocol deviations,
and negative examples together. Keep verbatim comments private unless the
participant explicitly approves publication. Until all 10 sessions are complete,
the repository and portfolio must continue to say that no human preference
evidence exists.
