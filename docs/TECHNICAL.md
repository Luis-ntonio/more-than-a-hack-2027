# pAEA · Ball possession from automatic tracking, trained only on external data

Team **pAEA** · More than a Hack 2027, Phase 1 · submission `pAEA.csv`
(SHA-256 `7d52e2acdc3ede04a49c6294800feba511fa0a5adfb7907e4a2b8b87db85f75f`)

## 1. Approach in one paragraph

Two gradient-boosted models read per-frame features of the tracking: one says whether the ball is
dead, the other which team has it. A three-state hidden Markov model turns their frame-by-frame
probabilities into whole possessions, which is what `spellF1` rewards. Everything is trained and
validated on **external, openly licensed matches**. The provided match only enters at the end, as
input, to produce the submission. We made this choice after the organisers' clarification of
3 November 2026: no number measured on the provided match, its video or its public ground truth sets
any parameter of the pipeline, and no hand labels are used anywhere.

## 2. Pipeline

```
External data (training)                         Provided match (input only)
------------------------                         ---------------------------
IDSSE: 7 Bundesliga matches, optical tracking    ATD tracking: players + ball (ball missing on 41%)
SkillCorner: 20 A-League matches, broadcast      Tactical video -> ball detector -> gap filling
        |                                                  |
degradation of IDSSE (lost ball, noise,          same 169 features per frame
lost players, simulated detector fill)                     |
        |                                                  |
169 features per frame  --->  [dead/alive model] [team model] [3-state HMM]  --->  pAEA.csv
```

**Ball enrichment** (`enrich/`). Where the ATD lost the ball, the
[roboflow/sports](https://github.com/roboflow/sports) YOLO ball detector (MIT) runs on the tactical
video in 640 px tiles and keeps the top-5 candidates per frame, projected to the pitch with the
provided homography. A candidate becomes the ball only if it is reachable at a physical speed
(40 m/s plus 2 m) from the last accepted position, starting from the last ATD position before the
gap. Accepted runs shorter than 3 frames are dropped unless every frame has confidence of at least
0.60, and holes of up to 5 frames are interpolated. Every threshold was chosen on **SoccerTrack v2**
video (CC BY 4.0), not on the match. There, 88.9% of the frames we fill are within 5 m of the ball.
A control run with unrelated detections falls to about 5%, so the precision comes from the detector,
not from the anchor.

**Features** (`external/features.py`). There are 149 base columns, computed from the home side's
point of view with x pointing to the goal home attacks:
- distance of each team to the ball, players within 3 m and 6 m;
- ball speed, acceleration, turn and position;
- last and next touch;
- team centroids and pace;
- means over 0.5 to 5 s windows, centred, past and future.

The team model also gets 20 attack columns: the speed of the ball and of the players' block towards
the opponent's goal. A `filled` flag marks frames whose ball came from the detector.

**Degradation** (`external/degrade.py`). IDSSE is clean optical tracking, so it is degraded before
training:
- the ball is lost for 10 to 70% of frames, more often in crowds and in the air;
- position noise is added, and players disappear;
- part of the lost frames get a simulated detector fill, both right and wrong (5 to 25 m off).

Levels are drawn per 5-minute stretch over wide ranges, so the model is trained across them instead
of on one assumed level. The 41% published for the match was deliberately not used.

**Models** (`external/train.py`):
1. **Dead/alive:** LightGBM on the 149 base columns, IDSSE only.
2. **Home/away:** LightGBM on all 169 columns, on IDSSE alive frames plus SkillCorner at 0.10 of
   IDSSE's weight.
3. **Decoding:** a 3-state HMM, Viterbi per half. Its transition costs come from IDSSE label counts.

Every match is used twice, the second time with home and away swapped. Every 5th frame is kept.
Everything is seeded and deterministic.

## 3. Assets

| asset | licence | used for |
|---|---|---|
| ATD tracking, homography, tactical video of the match | challenge data | input only |
| IDSSE / DFL tracking and `BallPossession` / `BallStatus` (Bassek et al. 2025) | CC BY 4.0 | training, validation |
| SkillCorner open data, 20 matches | MIT | team model training, validation |
| SoccerTrack v2 panorama video and events | CC BY 4.0 | choosing detector crop and gap-filling thresholds |
| roboflow/sports `football-ball-detection.pt` | MIT | ball candidates in tracking gaps |
| PBDet-Soccer | CC BY 4.0 | fine-tuning attempt, dropped |

Every external download is checked against a pinned SHA-256 (`external/downloads.sha256`).

**Tried and dropped.** Each idea was judged on held-out external matches against a rule written down
before running it.

| idea | result on held-out external data | decision |
|---|---|---|
| Smart Tagging as input, tags simulated from external truth at 4 noise levels | +0.02 to +0.10 spellF1 with decent tags, −0.005 with useless ones; trust columns did not fix it | dropped: its real error rate cannot be measured without the match |
| semi-Markov duration decoding | fitted durations collapse to the plain HMM, +0.001 | dropped |
| continuity and shape features | no gain, or a loss on SkillCorner | dropped |
| SkillCorner in the dead/alive model | learns another provider's dead-ball definition | dropped |
| detector fine-tuned on PBDet | 1.4% vs 7.0% of top candidates within 5 m on panorama | original weights kept |
| tracking the ball from both ends of a gap | +0.1 point filled within 5 m, same as control, −3.6 points precision | dropped |

## 4. Possession rules

We implement the challenge's definition by learning it from a provider whose labels have the same
structure:
- **Dead ball → `D`.** IDSSE's `BallStatus` marks the referee's stoppages, so the dead/alive model
  learns dead balls, restarts and the time before kickoffs. Who will restart a dead ball is not
  predicted, as the challenge asks.
- **Loose or contested ball → the last team in possession.** DFL's `BallPossession` is never empty and
  keeps the last team through contests, as `CHALLENGE.md` describes. The HMM enforces this
  structurally, because changing team costs more than staying.
- **Changes of possession** happen where the team model's evidence outweighs the HMM's cost of
  switching. The cost comes from how often possession changes in IDSSE: about 4 to 5 times a
  minute, with 30% of possessions shorter than 2 s.
- **Kickoffs.** Graded frames before each half's kickoff anchor in `games.py` are `D`. Frames
  outside the graded blocks are `X`.

## 5. How well it works, on external data only

The provided match is never scored, so these figures come from matches the models did not see. The
organisers' baseline algorithm (`starter/baseline.py`: nearest player, `D` when the ball is lost,
±50-frame majority vote) is run on the same frames, reading the raw tracking without the fill, as on
the match. The models here are trained on 5 IDSSE matches only, so all 20 SkillCorner matches are
held out.

| held-out data | baseline ±50 | ours |
|---|---|---|
| IDSSE, 2 matches, clean | 0.293 | **0.590** |
| IDSSE, same matches degraded, 3 seeds | 0.192 | **0.568** |
| SkillCorner, 20 matches, real broadcast tracking | 0.251 | **0.361** |

The figures are spellF1. SkillCorner is the most realistic test, because its tracking is automatic
and its definition of possession is another provider's. On it, team assignment agrees with the truth
on 75 to 78% of alive frames.

## 6. Limitations

- **Domain gap.** The largest unknown is how far the provided tracking is from our simulated
  degradation. Measuring it would mean tuning on the match, which the rules forbid.
- **Definitions differ by provider.** DFL, SkillCorner and the challenge's ground truth may draw
  ownership of loose balls differently.
- **Team assignment** is the weakest part.
- **Ball fill is sparse.** On SoccerTrack it fills only about 10% of hidden frames, though precisely.
  A tactical camera should do better, but by how much cannot be measured here.
- **Possessions get merged.** On IDSSE we predict fewer possessions than there are (about 320 vs
  412), and each merged possession costs spellF1.
- **With more time** we would try Smart Tagging with a measured error rate, for example from another
  match with both tags and ground truth, and a degradation learned from SkillCorner's real tracking
  gaps.

## 7. Reproduction

Requires Docker and the data bundle unpacked into `data/`. The GPU image is only needed for the ball
detector.

```bash
docker compose build dev && docker compose build gpu
docker compose run --rm dev python -m external.idsse download        # ~2.6 GB, hash-checked
docker compose run --rm dev python -m external.idsse parse
docker compose run --rm dev python -m external.skillcorner download  # ~1.8 GB, hash-checked
docker compose run --rm dev python -m external.skillcorner parse
docker compose run --rm gpu bash enrich/download_models.sh           # detector weights
docker compose run --rm gpu python -m enrich.detect_ball             # ball candidates in ATD gaps
docker compose run --rm dev python -m enrich.merge                   # enriched ATD
docker compose run --rm dev sha256sum -c enrich/artefacts.sha256     # matches the pinned files
docker compose run --rm dev python -m external.train --final
docker compose run --rm dev python -m external.predict               # evaluation/results/external.csv
docker compose run --rm dev python evaluation/validate_submission.py evaluation/results/external.csv
```

From a fresh clone with checked downloads, this regenerates the submission bit for bit (SHA-256
above). We checked it on a fresh clone: the ball detections, the enriched ATD and the final CSV all
match their pinned hashes. The detailed log of every decision and its numbers is in
[`docs/07_entrenamiento_externo.md`](07_entrenamiento_externo.md), in Spanish.
