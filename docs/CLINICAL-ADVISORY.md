# Clinical and participatory advisory model

Mekasa is an experimental assistive-technology research prototype, not a medical device. This document describes collaboration needs; it does not provide medical advice or authorize human research.

## Decisions requiring participatory and specialist review

- whether gaze is a useful, comfortable and chosen access method for a particular person;
- vocabulary, symbols, board organization, confirmation and repair strategies;
- positioning, mounting, viewing distance, session duration and fatigue responses;
- visual contrast, target spacing, calibration behavior and ocular-motor access needs;
- fallback communication and what happens during tracking/TTS failure;
- study eligibility, consent, stopping criteria, outcomes and interpretation;
- claims about functional communication, usability, safety or clinical benefit.

AAC users are design participants whose goals and experience take priority in judging usability. Depending on the question, collaboration may include family/caregivers, an AAC-specialist SLP/fonoaudiólogo, occupational therapy, PM&R, physiotherapy, vision/ophthalmology, rehabilitation engineering and accessibility/HCI.

## Primarily technical decisions

Dependency locking, process boundaries, serialization, checksums, test isolation, memory handling and benchmark instrumentation are engineering responsibilities. Their consequences still need privacy, accessibility and security review. Dwell/confidence/calibration thresholds are not merely implementation constants when they affect effort or unintended communication.

## Feedback process

Before involving AAC users, define an accessible feedback method, consent/withdrawal path, fallback communication method, break/stop procedure, data inventory and responsible reviewer. Record dissent and failed sessions, not only successful runs. Technical reports must separate observed facts, participant feedback, specialist interpretation and engineering inference.

Open questions are maintained in `V0-CLINICAL-QUESTIONS.md`. They remain open until addressed with appropriate people and evidence.

