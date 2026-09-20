# Mekasa V0 — Clinical and Participatory Design Questions

Status: open questions; no clinical answers or recommendations are asserted  
Date: 2026-09-19

Mekasa is an experimental assistive-technology research prototype, not a medical device. These questions must be discussed with relevant AAC users and specialists. A diagnosis alone must never be used to infer suitability. Participation, consent, accessible communication, the right to stop and the AAC user's own priorities are prerequisites.

## Questions for AAC users

- Which four intentions are most valuable in the person's real contexts, and is a 2×2 board understandable?
- How should the system show that it has detected, is dwelling on, selected or lost a target without increasing visual fatigue?
- What should pause, cancel, undo, repeat and help look like when they must be available through eye-only access?
- What dwell duration feels controllable rather than slow or accidental, and how should the person change it?
- What kinds of unintended selection feel most disruptive or distressing?
- What posture, distance, screen position, lighting, glasses or mounting arrangements are comfortable?
- How should calibration explain failures without implying user fault?
- What information and controls should remain private from caregivers or other people nearby?
- What outcomes represent success to the AAC user: reliability, speed, reduced effort, autonomy, being understood, or something else?
- How and when should the user be able to withdraw from research data collection or delete contributed data?

## Questions for a physiatrist / PM&R specialist

- Which functional characteristics, rather than diagnoses, should prompt an individualized access assessment?
- What changes in fatigue, positioning, alertness or motor control may affect session duration or repeatability?
- What contraindications or stop criteria should a formal evaluation protocol consider?
- How should the team distinguish technical tracking failure from changes that merit clinical review, without the software making a diagnosis?
- Which disciplines should coordinate positioning, access method and follow-up for a particular participant?
- What claims or instructions would incorrectly imply medical-device status or clinical effectiveness?

## Questions for an SLP / fonoaudiólogo/a specializing in AAC

- Is YES/NO/WATER/HELP an appropriate experimental mapping, and how should vocabulary be selected for an individual?
- How can the board preserve semantic clarity across languages, literacy levels and symbol systems?
- Which communication metrics reflect functional value beyond target-selection accuracy?
- How should confirmation, repair, rejection, repeat and conversational turn-taking work with only four targets?
- When could dwell-based selection interfere with visual exploration, language formulation or partner interaction?
- What training is appropriate for the AAC user and communication partners without conflating training effects with system accuracy?
- How should existing AAC methods remain available as fallback and not be displaced by an experimental prototype?

## Questions for occupational therapy

- What seating, mounting, camera/screen position and viewing distance are sustainable and repeatable?
- How should setup account for involuntary movement, asymmetry, range of motion, glasses and environmental lighting?
- Which signs of visual, cognitive or postural fatigue should cause a pause or end a session?
- How should caregiver/setup controls be arranged so they do not compromise the user's autonomy?
- Which alternative access methods (switch, touch, keyboard, partner-assisted scanning) should coexist for safety and choice?
- What setup variability should be recorded to interpret technical measurements responsibly?

## Questions for a vision / ophthalmology specialist

- Which visual acuity, field, fixation, ocular-motor, diplopia, nystagmus, ptosis, eyelid, pupil, eyewear or photosensitivity factors may affect feasibility?
- What observations can a non-clinical prototype safely request without attempting screening or diagnosis?
- Are the target size, contrast, spacing and calibration motion suitable for the individual's visual access needs?
- How should monocular use, unequal eye visibility or fluctuating tracking be represented without assuming failure?
- What limits should apply to screen brightness, animation, fixation duration and session length in a research protocol?
- When should a technical team defer to a qualified vision evaluation?

## Questions for a rehabilitation engineer / assistive-technology specialist

- Which commodity cameras, mounts, displays and computing devices represent realistic target environments?
- What camera placement and field of view tolerate real-world changes without restricting the user's posture?
- How can setup be repeatable while remaining adjustable and transportable?
- Which external eye trackers or infrared cameras should later implement the same `CameraProvider`/`GazeEstimator` boundary?
- How should device compatibility, failure modes, cable/power issues and local-process security be documented?
- Which acceptance tests demonstrate that the system fails safely when tracking or TTS is unavailable?

## Cross-disciplinary protocol questions

- Who decides whether an evaluation is appropriate, and how is the AAC user's preference given decisive weight?
- What accessible consent/assent process and research/ethics review are required for each study context?
- What is the comparator: current communication method, partner-assisted method, commercial eye tracker or within-person baseline?
- How many sessions and environmental conditions are needed before describing repeatability?
- How will breaks, early stopping, missing data and failed calibration be reported rather than hidden?
- Which outcomes are purely technical, which are usability outcomes and which require clinical interpretation?
- How will selection errors be repaired during a session so they do not create false communication?
- How will subgroup performance and exclusions be reported without implying generalization?
- What minimum evidence is required before moving from developer self-testing to studies with AAC users?
- How are adverse experiences, privacy incidents and accessibility barriers reported and acted upon?

## Decisions that remain technical, with stakeholder review

Implementation details such as process boundaries, serialization, dependency locking, checksums, test architecture and frame-memory handling are engineering decisions. They still require privacy, accessibility and security review when their consequences affect users. Thresholds for dwell, tracking confidence, calibration acceptance and session behavior are not purely technical constants; they require measured evidence and participatory review.

## Evidence log to create before human research

- approved study purpose and review/oversight determination;
- participant-facing accessible information and consent materials;
- inclusion/exclusion rationale based on access needs rather than convenience;
- data inventory, retention, deletion and access controls;
- stopping, pause, fallback communication and incident procedures;
- device/setup description and versioned software configuration;
- preregistered technical and functional outcomes where appropriate;
- results including failures, missing data and limitations;
- a plain-language feedback path for participants.

None of these items authorizes a study. They identify work that must be completed with qualified people and applicable institutional/legal oversight.
