# Two Models, One Verdict: Building SonicSentinel AI, a Sound-Event Detector That Knows When It Is Unsure

*A technical write-up of our TechWiz 7 project (NextWave AI and ML · AcousticX Intelligence).*

---

## The business problem

A gunshot, breaking glass, a scream, a person shouting "help me", a siren, a machine that has started to
grind — each of these sounds means someone should do something, usually quickly. In most buildings, plants
and public spaces today the only "sensor" for them is a person: a guard listening to a live feed, or an
engineer scrubbing through a recording after something already went wrong.

That approach has three well-known problems. It does not scale — one operator can follow only a handful of
streams. It is inconsistent — two people judge the same muffled bang differently, and the same person judges
it differently at 3 a.m. And it fails exactly where it is needed most, in noisy, crowded places where the
important sound is buried under traffic, rain or conversation.

SonicSentinel AI is our answer: a web application that listens to uploaded recordings and live microphones,
recognises ten kinds of sound events, and raises alerts with a severity and a recommended action. Just as
important, it tells the operator *how sure it is* and sends every doubtful case to a human reviewer instead
of guessing.

## Background: what "detecting a sound" really involves

A sound-event detector has to answer three questions for every short piece of audio: *what* is it, *how
confident* are we, and *does it matter*? The first is a classification problem. The second is harder than it
looks, because a classifier's probability is only as honest as its training data. The third is a policy
question — a car horn is information, a gunshot is an emergency — and it should be configurable by the people
who run the site, not hard-coded by developers.

The SRS for this competition adds a twist that shaped the whole design: every sound must be classified by
**two independently trained models** — a Python machine-learning model and a Google Teachable Machine (GTM)
audio model — and the app must compare them. We turned that requirement into the core idea of the system:
two models that were trained differently and look at the audio differently act as a second opinion for each
other.

## Proposed solution

The application supports two input modes: file upload (single files or batches of up to 50) and live
microphone monitoring in 1–3 second windows. Every input goes through the same path: validation, metadata
extraction, pre-processing, segmentation into 2-second windows, audio-quality analysis, classification by
both models, comparison, the critical-event rule engine, and finally storage, dashboards and alerts.

There are five roles. A normal user uploads and analyses audio. An audio reviewer works through the
manual-review queue. A security operator and a maintenance operator each get a console with the alerts for
the categories their role is responsible for. An administrator configures rules, thresholds, users, models
and data retention.

## Architecture

The backend is Flask with SQLAlchemy (SQLite for the demo, PostgreSQL-ready). Audio work uses NumPy, SciPy,
soundfile, librosa and FFmpeg as a fallback decoder. The Python classifier uses scikit-learn and XGBoost. The
GTM model is a TensorFlow.js model that runs in the user's browser.

The flow for one uploaded file is:

1. **Validate** — the file content is sniffed (not just the extension), then size, duration, sample rate,
   channel count, decoding integrity and the presence of a real signal are checked. A WAV file whose header
   promises more audio than it contains is flagged as truncated.
2. **Pre-process** — resample to 22,050 Hz, mix to mono, peak-normalise, trim leading and trailing silence,
   and apply adaptive noise reduction.
3. **Segment** — 2-second windows with a 1-second hop; start and end times are stored for each.
4. **Quality** — silence, clipping, noise floor, low signal, duration, encoding problems and dropouts give a
   label: Good, Acceptable, Poor or Unusable.
5. **Classify** — the server extracts 193 features per segment for the Python model; the browser computes a
   spectrogram of the same segment for the GTM model.
6. **Compare and decide** — the rule engine combines both models, quality and history into a final decision.

For live monitoring the browser records windows with an AudioWorklet, runs GTM on each window, and posts the
window plus the GTM scores to the server, which does the rest and returns the result in well under the
three-second budget.

## Dataset collection

The SRS asks for at least 3,000 unique original clips, about 300 per class. Our dataset has **7,866 original
clips** plus 5,190 augmented copies, 7.08 hours in total, split 70 / 15 / 15 into training (5,506),
validation (1,179) and test (1,181) recordings. It covers the 10 mandatory categories **and all 7 optional
ones** the SRS lists — Drone Sound, Drilling or Grinder Sound, Fireworks, Vehicle Backfire, Explosion, Door
Impact and Graffiti Spray — 17 classes in total.

They come from three sources:

* **UrbanSound8K** (2,316 clips, CC-BY) — real field recordings of gun shots, sirens, car horns, dog barks,
  drilling, jackhammers, idling engines, air conditioners, street music and children playing.
* **Procedural synthesis** (5,135 clips, CC0) — our own generators for glass breaking, gunshots, screams,
  alarms, machinery faults, drones, fireworks, backfires, explosions, door impacts and spray cans. Each clip
  is rendered with a random device (smartphone, laptop, headset, body camera, CCTV, studio recorder),
  distance (0.5 to 30 m), room (small room, hall, factory, street, open field), reverberation, background
  (rain, wind, HVAC, traffic, crowd, conversation) and a signal-to-noise ratio between 0 and 35 dB. Some
  clips contain a second, quieter overlapping event.
* **Text-to-speech** (415 clips, CC0) — the help phrases, aggressive shouting and calm everyday sentences,
  spoken by several synthetic voices (male, female, child) at different speeds and pitches.

Every clip has an Audio ID and a metadata row: class, split, source, licence, duration, sample rate,
channels, bit depth, environment, device, distance, background, SNR, overlap, speaker, phrase, and whether it
is an original or an augmented copy. Augmented copies keep the Audio ID of their original and always stay in
the same split, so no version of a test recording is ever seen during training.

## Ethical sourcing

We did not record anyone in real distress, and we did not scrape audio of unknown origin. The "Person Asking
for Help" class is limited to the defined safety phrases — "Help me", "Somebody help", "Please help", "Call
for help", "Emergency" — generated with text-to-speech. Real recordings come only from a dataset with a clear
licence (CC-BY), and generated audio is released as CC0.

## The day the labels turned out to be wrong

Our test accuracy said 83.4 %. Then we started feeding the app sounds that were not from our dataset at all —
files downloaded from a sound library — and it fell apart in ways the test split had never shown:

* a 60-second recording of an **empty room** came back as Machinery Fault;
* a **camera shutter** came back as Gunshot with 94 % confidence;
* a **horse neigh** came back as Unknown Sound;
* a **vehicle horn** split the two models: Python said Person Asking for Help, GTM said Gunshot.

None of these were model bugs. They were all data problems, and the test split could not see them because it
contained the same problems.

The empty room was the clearest. We had imported UrbanSound8K's `air_conditioner` and `engine_idling` classes
into **Machinery Fault**, reasoning that they were machine sounds. They are machine sounds — but they are
*healthy* machine sounds, steady ambient hum, which is what our Background Noise class is for. We had taught
the model that any steady hum is a fault. That one mistake also explains why Machinery Fault was our weakest
class at 0.44 recall: the class contained two incompatible ideas of the sound, and 27 of its 45 test errors
were confusions with Animal Sound, because engine harmonics and animal growls look alike in MFCCs once you
have told the model that both are "machinery". We moved those 150 clips to Background Noise.

The camera shutter was the opposite problem: not a wrong label, but a missing one. Our dataset contained no
everyday impulsive sounds that are *not* events. Every sharp click the model had ever seen was a gunshot,
glass breaking or a door slam, so a sharp click it had never seen became the nearest of those. We added
camera shutters, light switches, keyboards, pens, staplers and mouse clicks as Background Noise, along with
quiet empty rooms (mains hum, low rumble, hiss) and calm neutral speech.

The horse taught us the same lesson about the positive classes: Animal Sound contained only dogs, cats,
birds, cows, roosters, crows, sheep, frogs and ducks. No large animals. We added horse neighs, big-cat roars,
wolf howls and elephant trumpets.

Finally, UrbanSound8K's `drilling` and `jackhammer` were also in Machinery Fault. They are working power
tools, and the SRS already lists "Drilling or Grinder Sound" as an optional category — so we implemented
that category, moved those clips into it, and imported 420 more from as many different source recordings as
possible. That decision led to implementing all seven optional categories.

The result of the relabelling alone, with no change to the algorithm: **Machinery Fault recall went from
0.444 to 0.889.**

The lesson we would give to anyone doing this: your test split is drawn from the same data as your training
split, so it cannot tell you that your labels are wrong. Only sounds from outside your dataset can.

## Dataset challenges

The honest headline is that **71 % of our originals are synthetic** — higher than before, because all seven
optional categories are generated. Glass Breaking, Panic Scream and every optional class are entirely
synthetic, and Person Asking for Help is entirely synthetic speech. A synthetic scream has a clean,
predictable structure; a real one does not. We fight this with heavy variation in device, distance, room and
background, but real-world accuracy on these classes will be lower than our test numbers.

Machinery Fault is now the clearest gap. The relabelling was correct, but it removed every real recording
from that class: what remains is 300 generated clips of grinding bearings, belt squeal, knocking and
imbalance. The class is coherent now, which is why its recall doubled, but it has never seen a real faulty
machine. That is the first thing we would record if we had access to a plant.

## Teaching the model what a microphone sounds like

There is a second gap that accuracy numbers hide completely. An uploaded file and a live microphone window do
not reach the model in the same shape:

| | Upload | Live window |
|---|---|---|
| Silence trimming | yes, whole file | never |
| Peak normalisation | over the whole recording | per 2-second window |
| Gain control | none | the browser's AGC |
| Frequency response | whatever the file has | the microphone's band limit and self-noise |

We had trained the Python model only on the upload rendering. Every number we reported was therefore a number
for the upload page, and the live monitor — the feature the SRS cares most about — was running on audio the
model had never been shown.

`feature_extraction/live_render.py` fixes that. It takes a dataset recording and puts it through the live
chain: one of four microphone profiles (band limit, self-noise, presence peak), a random distance
attenuation, an automatic gain control that follows the level the way a browser's does, resampling to
22,050 Hz, then per-window pre-processing with no trimming — exactly what `live_service.process_window` does
with a real window. A test asserts that a rendered window is byte-identical to what the live service
computes, so the two can never drift apart.

Every training recording now exists in both renderings. The models are still selected and reported on the
upload rendering, so the headline numbers stay comparable, and the live rendering is evaluated separately and
reported next to them. For the GTM model, which already trained on both, the live-microphone path scores
79.8 % against 80.1 % for uploads — within a rounding error, which is the point.

## Audio pre-processing

Pre-processing is shared by training and prediction, so the model always sees audio prepared the same way.
The step we spent most time on was noise reduction. We use spectral gating: the quietest 15 % of frames
define a noise profile, and time-frequency bins that are not clearly above it are attenuated by at most
12 dB with a smoothed mask. The catch is that for a *stationary* sound — a siren, a machine hum — the
"quiet" frames are the sound itself, so the gate would remove the very sound we want to classify. The
algorithm therefore measures the recording's dynamic range first and skips noise reduction when it is below
12 dB.

## Feature extraction

For each 2-second segment we compute 193 features: 20 MFCCs (mean and standard deviation) and their deltas,
32 mel-band summaries, 12 chroma bins, 7 spectral-contrast bands, spectral centroid, bandwidth, roll-off,
flatness and zero-crossing rate, RMS energy statistics (including dynamic range and crest factor), onset
strength and onset rate, tempo, pitch statistics (median, spread, stability), harmonic and percussive ratios,
and temporal shape features — where the peak is, how fast the attack is and how fast the sound decays.

The last group is small but useful: a gunshot has a near-instant attack and fast decay, glass breaking has a
fast attack and a long ringing tail, and a siren has almost no attack at all.

## Python model development

We tuned five algorithms with a randomised hyper-parameter search, using group k-fold cross-validation
grouped by the original Audio ID — otherwise segments and augmented copies of one recording would sit on both
sides of a fold and inflate the score. Scoring was macro F1, so small classes count as much as large ones.

The tuned models were compared on the validation split, at recording level:

| Model | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---|---|---|
| SVM (RBF kernel) | 0.859 | 83.4 % | 0.841 |
| Random Forest | 0.866 | 81.2 % | 0.814 |
| Histogram Gradient Boosting | 0.883 | 82.5 % | 0.833 |
| Multi-layer Perceptron | 0.884 | 83.8 % | 0.843 |
| XGBoost | 0.886 | 84.2 % | 0.853 |
| Soft-voting ensemble (XGBoost + MLP + HGB) + critical weighting | **0.895** | **83.4 %** | **0.844** |

The ensemble won on validation and was selected. We deliberately did **not** pick the model with the best
*test* score (XGBoost), because choosing on the test set would turn it into a second validation set and make
the reported number meaningless. The price of that discipline is visible: our Python model reaches 83.4 %
test accuracy, short of the 85 % target, and its validation-to-test drop (0.895 to 0.844 macro F1) shows the
selection over-fitted the validation split a little.

We also added critical-class weighting: after training, the probabilities of Gunshot, Panic Scream, Person
Asking for Help and Aggression are multiplied by weights tuned on validation (2.5, 1.75, 1.5 and 1.25). This
trades a little precision on those classes for fewer missed emergencies.

## GTM training

Teachable Machine audio models are built on Google's *speech-commands* network: the browser turns one second
of 44.1 kHz audio into a 43 × 232 log-magnitude spectrogram (0–5 kHz) and a small convolutional network
classifies it. Teachable Machine trains only a new output layer on top of that frozen network.

Our first GTM export scored **38 %** in the app. Two bugs explained most of it: the training samples were
always the *first* second of each file, while the browser classifies the *loudest* second; and training used
raw files while the app gives GTM pre-processed audio. We rewrote the GTM input path in Python so it
reproduces the browser bit for bit, and checked it against 33 results recorded by the real browser — the
same window was chosen every time.

We then compared three variants on validation: the Teachable Machine "transfer" setup (frozen base, new
output layer: 77.9 % test accuracy), the same network fine-tuned end to end (86.5 %), and fine-tuned with
critical-class weighting (86.5 %, selected). The exported TF.js files were verified against the trained
network with a NumPy re-implementation (maximum difference 7.5 × 10⁻⁷). GTM reaches 86.5 % test accuracy,
macro F1 0.871, and at least 91 % recall on every critical class — it meets all the SRS accuracy targets.

## Models compared, and confidence comparison

For every segment the app stores both predicted classes, all ten confidences from each model, whether the
classes match, the absolute difference between the two top-class confidences, and each model's top-two
margin. From these it derives a consistency status:

* **Acceptable Match** — same class, both confidences at least 0.60, and within 0.25 of each other;
* **Weak Match** — same class, but one model is unsure or they differ a lot in confidence;
* **Model Disagreement** — different classes, both confident;
* **Uncertain Result** — different classes with low confidence, or unusable audio.

On our 100-recording comparison report (10 unseen test recordings per class), Python was right 89 times, GTM
92 times and the final fused decision 91 times. The models agreed on the class 87 % of the time, with a mean
confidence difference of 0.17. The distribution was 64 Acceptable Matches, 23 Weak Matches, 3 Model
Disagreements and 10 Uncertain Results.

The final decision averages the two probability vectors. When the fused confidence is below 0.40, the
result becomes "Unknown Sound" and goes to review rather than being forced into a class.

## Waveforms and spectrograms

Each analysed recording shows its waveform and a mel spectrogram. The waveform is sent to the browser as
min/max peaks per pixel column and drawn on a canvas, so it follows the colour theme and shows a play-head
that moves with the audio player. The spectrogram is a 128-band log-mel image rendered on the server in
grey-scale, loud as dark like a printed sonagram, and inverted in dark mode. The player also shows a
track of the detected events, and reviewers can play just the segment that triggered a result. For the PDF report, matplotlib versions with axes are embedded.

These views are not decoration: a reviewer can see at a glance that a "gunshot" is actually three evenly
spaced impacts from a nail gun, or that a "siren" is flat and has none of a siren's sweep.

## Alert rules

Alert behaviour lives in a JSON file that administrators edit in the UI; every save creates a new version and
an audit record. Each category has a severity, whether it raises an alert, a minimum confidence, a minimum
top-two margin, how many consecutive detections are required, whether both models must agree, the minimum
audio quality, when to send it to manual review, an escalation rule, the responsible roles, and a recommended
action.

The defaults follow the SRS: Gunshot, Panic Scream and Person Asking for Help are Critical; Glass Breaking,
Alarm or Siren, Aggression and Machinery Fault are High; Vehicle Horn and Animal Sound are Low; Background
Noise is Informational. A gunshot alert needs 0.70 confidence, a 0.20 margin, two consecutive detections,
agreement of both models and at least Acceptable audio. Aggression escalates from High to Critical after three
detections within 30 seconds. Background Noise becomes Medium, with an alert, only when the measured noise
floor exceeds the configured limit.

## Real-time monitoring

The live monitor shows the microphone state (available, active, paused, disconnected, permission denied), the
current sound, both models' results with their top three classes, the agreement status, severity and any
active alert, plus a live 3D spectrum of the microphone.

Repeated detection is where live monitoring differs most from uploads. After the first gunshot window the
alert status is *Pending Confirmation*; if the next window also says gunshot, the alert is generated. A
single transient bang — a door, a dropped box — rarely survives two windows and model agreement.

## Difficulties

* **The GTM window bug** (training on the first second, predicting on the loudest) taught us to test the
  exact path the app uses, not a convenient approximation of it.
* **Noise reduction versus stationary sounds** — solved with the dynamic-range check described above.
* **Leakage** — with 2,100 augmented copies and several segments per recording, ordinary k-fold would put
  versions of one recording on both sides of a fold and inflate the scores. Every split and fold is grouped
  by the original Audio ID to prevent this.
* **Truncated files** — the audio library silently reports a truncated WAV as a shorter, valid file. We now
  read the WAV header ourselves and compare the promised length with the actual length.

## False positives

Animal Sound has the lowest precision in the Python model (0.66): machinery and background clips are pulled
into it, because an engine's harmonics look like a growl in MFCC space. Gunshot precision is 0.84, mostly
impulsive machinery such as jackhammers. For alerts, false positives are held back by the rules: gunshots
need two windows, both models and decent audio. When an operator dismisses an alert, the event goes to the
review queue as a possible false alarm, so it can be corrected and later used to improve the data.

## False negatives

The misses that matter most are in Machinery Fault (Python recall 44 %): 27 test faults were called Animal
Sound, 14 Gunshot and 9 Background Noise. That is the class-definition problem described earlier. Among the
critical classes, Python misses 20 % of help requests (4 of them read as Aggression); GTM misses only 2 %.
Combining the models helps here: on the 100-recording report the final decision's critical-class recall was
94 %.

## Model disagreement

Disagreement is not an error to hide — it is information. In the comparison report the models agreed on only
60 % of Aggression clips (Python 70 % correct, GTM 90 %) and 80 % of Vehicle Horn, Animal Sound and Help
clips. Each disagreement is shown to the operator as "Model Disagreement" and routed to review. On the full
test set the two models make different mistakes — GTM confuses sirens with horns, Python confuses machinery
with animals — which is exactly why using both is worth it.

## Noise robustness

Noise was built into the data from the start: backgrounds at 0–35 dB SNR, simulated distance, reverberation
and six device types, plus augmentation with more of the same. The training script also evaluates every
model on test recordings with added noise at 20, 10, 5 and 0 dB SNR (results in each model's
`noise_robustness.json`). For GTM we measured the live path
separately — raw microphone audio without noise reduction — and it reached 87.6 %, slightly *above* the
processed path, which suggests the model did not learn to depend on our noise reduction.

## Security

Passwords are salted hashes; accounts lock after five failed sign-ins and failed attempts raise system
alerts. Every form and API call is protected by a CSRF token, every route checks the user's role, and alert
actions are limited to the roles responsible for that category. Uploaded files are identified by content,
stored under random names and served only through permission-checked routes. The Content-Security-Policy
allows no external scripts — every library is bundled with the app. An audit trail records logins, uploads,
microphone sessions, predictions, alerts, reviews, overrides, exports and every model or rule change.

## Privacy

The microphone never starts without a consent notice, and a visible indicator shows whenever it is on. By
default only live windows that contain an event are stored. Retention is configurable (uploads 30 days, live
audio 7 days, records one year) and a clean-up runs daily. The help-phrase class detects only a few
predefined phrases; the system does not transcribe speech or identify speakers.

## Limitations

* The Python model is at 83.4 % accuracy, below the 85 % target; Person Asking for Help (80 %) and Machinery
  Fault (44 %) recall are its weak points. GTM meets the targets.
* Most of the data is synthetic; hidden tests with real recordings will be harder.
* Overlapping sounds are detected and flagged, not separated.
* GTM runs in the browser; if it cannot load, the system falls back to the Python model and sends critical
  events to review.
* This is a competition prototype, not a certified safety system.

## Lessons learned

1. **Test the path you ship.** Our GTM model looked perfect in isolation and was 38 % in the app.
2. **Guard the test set.** It is tempting to pick whatever scores best on test; doing so would have given us
   a nicer number and a dishonest one.
3. **Two models are a feature.** Their disagreement found more doubtful cases than any single confidence
   threshold.
4. **Data beats tuning.** Every large gain came from fixing data (grouping, windows, class definitions),
   not from hyper-parameters.
5. **"Unsure" is a valid answer.** Routing uncertain events to a reviewer makes the system more useful, not
   less.

## Future enhancements

* Record real, consented examples of the weakest classes — machinery faults, help phrases, sirens — and
  split Machinery Fault from a new Normal Machinery class.
* Add a convolutional network on log-mel spectrograms to the Python ensemble.
* Train the optional classes (Fireworks, Vehicle Backfire, Explosion) to tell them apart from gunshots.
* Separate overlapping sources, and estimate direction with microphone arrays.
* SMS and e-mail notifications for critical alerts, and multi-site dashboards.

---

*SonicSentinel AI is a competition prototype for controlled and ethical testing. It is not a certified
emergency-response or law-enforcement system.*
