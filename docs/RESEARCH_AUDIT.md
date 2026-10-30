# SAC-RWB: evidence, reproduction audit, and research decision

Prepared for Neo, 29 October 2026. Discussion language: Hinglish.

## Decision at this stage

**Base paper eligible hai; research direction conditionally useful hai. Full RL training aur quantum novelty claim abhi premature hain.** Is session mein CPU par simulator/environment aur risk-predictor components execute hue. Published results reproduce nahi hue. Original upstream training/evaluation scripts have confirmed failures and algorithm/protocol discrepancies. Upstream tracked files have not been edited.

## Sources and provenance

- Attached final IEEE paper: Chengyang Dong and Nan Guo, *Biased-Attention Guided Risk Prediction for Safe Decision-Making at Unsignalized Intersections*, IEEE Transactions on Vehicular Technology 75(8), 16043–16054, August 2026, DOI 10.1109/TVT.2026.3680137. PDF first page records publication 2 April 2026. Therefore the 2025/2026 IEEE Transactions requirement is satisfied **according to the supplied published PDF**. IEEE/Crossref metadata was not independently retrieved in this session. The supplied arXiv v1 was not used as a substitute for the final PDF.
- Author repository: https://github.com/hank111525/SAC-RWB , exact audited HEAD `f60cdbadc8b65cf4699edc6566af7936ff02fba6`, cloned into `/workspace/research/SAC-RWB`. No release tags were returned. Latest local commit date is September 2025, preceding the final publication; this checkout should not be assumed to match the final manuscript exactly.
- **Attachment identity correction:** `Out-of-Distribution_Object_Detection_in_Autonomous_Vehicles_With_Yolo_Model.pdf` actually contains Dawei Wang, Weizi Li, Lei Zhu and Jia Pan, *Learning to control and coordinate mixed traffic through robot vehicles at complex and unsignalized intersections*, International Journal of Robotics Research 44(5), 805–825 (2025), DOI 10.1177/02783649241284069. Its content, not filename, establishes identity. No YOLO/OOD detection paper has been assessed.
- The other attachment is Kurt Dresner and Peter Stone, *A Multiagent Approach to Autonomous Intersection Management*, JAIR 31, 591–656 (2008).
- Wang is reference [21] and Dresner–Stone is [30] in the base paper, matching Table II. They are comparison sources, not alternative eligible base papers: IJRR is SAGE and the JAIR paper is from 2008.

Extracted text is under `papers/`. The base paper's pages 8 and 9 were also rendered and visually inspected to read Algorithm 1 and Tables II–III, which were absent from text extraction.

## What the base method actually does

1. SUMO models intersection traffic. A shared continuous SAC policy controls longitudinal acceleration of selected approaching vehicles within 30 m. Right-turn decisions are excluded from the controlled task. Manoeuvre/route categories are largely established by lanes; this is not an ego car inferring arbitrary hidden human intentions.
2. The 98-dimensional input comprises distance, speed, eight queues, eight waiting times and 80 occupancy bins. These are simulator-derived traffic features, not raw camera/LiDAR inputs. The action is one scalar acceleration in [-3,3] m/s². The configured simulator step is 1 s.
3. A risk Transformer consumes ten historical state–action pairs. Its additive attention bias is `B[i,j] = 0.2*(j-9)`, identical across query rows: a preference for newer keys. It is not itself a posterior over uncertain driver intentions, and it is not identical to full relative-distance ALiBi.
4. The predictor uses embedding dimension 128, two encoder layers, four heads and a 128–64–1 classification head. It is trained using safe and collision-labelled sequences sampled 50:50 from separate buffers. This hierarchical replay is for the predictor, not SAC's replay; it does not refer to hindsight experience replay.
5. Risk is used to shape SAC training rewards. In the code, `reward = efficiency - 3*predicted_risk + safety_term`. The paper's positive risk-weight notation needs reconciliation with its collision=1 labels; do not implement positive collision reward by copying Eq. (5) mechanically.
6. **During evaluation the paper disables the predictor and runs only the actor.** Thus this is primarily risk-guided policy learning, not an online uncertainty-aware safety controller.

Published training: 500 episodes × 1000 steps; actor/critic widths 256–256; SAC batch 256 and replay 1,000,000; actor LR 3e-4, paper critic LR 4e-4; risk batch 128, LR 1e-5 and two 10,000-sample buffers. Reported hardware is RTX 4060 Ti 16 GB.

## Published claims versus evidence available here

Table II in the final supplied PDF:

| Method | AQL (paper labels veh/s) | AWT (s) | CR (%) |
|---|---:|---:|---:|
| FCFS [30] | 8.52 | 199.26 | 0 |
| Wang [21] | 6.41 | 61.57 | 0.19 |
| SAC-RWB | 4.59 | 50.81 | 0.10 |

These are **authors' reported numbers**, not reproduced results. The SAC method does not improve collision rate over FCFS's reported zero; it exchanges some safety for efficiency relative to that comparator. Queue length is normally a vehicle count, so the table's veh/s label should be clarified.

Table III gives different absolute SAC-RWB values: AQL 3.24, AWT 21.08 s, CR 0.061%. Text associates it with 1000-step tests and three seeds. Table II says 1500 test rounds, while the nearby figure spans 1500 steps. The repository evaluation uses a single rollout and a default 500:1000 reporting window. These results must not be pooled or treated as one fully specified protocol.

The paper explicitly calls an in-intersection speed below 0.1 m/s a training "collision"/conflict and removes the vehicle, but disables that criterion at test time. This proxy is **not a physical crash**. Training conflict curves and test collision rates answer different questions. Report stopped-in-intersection events and actual collision events separately.

## Repository audit: dependencies, assets, and correctness

All source locations below refer to the immutable audited checkout above. Static findings and executed failures are distinguished.

| Finding | Evidence | Consequence |
|---|---|---|
| Requirements unsatisfiable | `requirements.txt` simultaneously pins Pillow 7.0.0 and 11.3.0; resolver dry run failed | Original documented install cannot succeed as written |
| Further version conflicts | PyPI metadata: Pillow 11.3 requires Python >=3.9; SB3 2.6 requires Python >=3.9 and Torch >=2.3; repo asks Python 3.7 and Torch 1.13.1+cu117 | Removing one Pillow line is insufficient |
| SUMO Python/binary mismatch | README recommends simulator 1.12.0, requirements pin traci/sumolib 1.23.1 | Align versions for a reference environment |
| Extraneous dependencies | SB3/roboschool not used by selected SAC path; RLlib is imported by Env for a typing alias | Audit actual imports instead of installing every unrelated package |
| Missing explicit TensorBoard requirement | `train_sac.py:13` imports SummaryWriter | Add in external environment lock; preserve upstream manifest |
| Wrong actor action dimension | `train_sac.py:101` sets action_dim=env.n_obs (98), while `Env.n_action` is 1 | Runtime actor test returns [3,98], not [3,1]. Flatten+zip silently misassigns/discards actions; replay scalar broadcasts across 98 entries |
| Every replay item terminal | `train_sac.py:207` passes True for every `dw`; `sac/SAC.py:48` multiplies bootstrap by ~dw | Bellman continuation is always zero; this is not faithful long-horizon SAC |
| Checkpoint save crashes | `train_sac.py:238` uses undefined opt.steps | Executed short run reached first save and raised AttributeError; default save would hit same expression |
| Evaluation parser crashes | `test_sac.py:47` uses type="str" instead of callable str | `test_sac.py --help` executed and raised ValueError |
| Further evaluation blockers | `test_sac.py:81` constructs SAC without populating state_dim/action_dim; max_action also unset; lines 83–85 hard-code run 36/step 500 | Fixing parser alone is insufficient; referenced weights absent |
| Training conflict proxy disabled | `Env.py:645–652` low-speed collision/removal logic commented out | Current training differs from a central paper setting |
| Efficiency reward discrepancies | `Env.py:366–367`: rank minus 5.5; paper subtracts mean rank; `:766–768` slices [12:20], whereas constructed waiting features are [10:18] | Reward semantics need verification, including ties; cannot silently change and call exact reproduction |
| Different normalization/LR | Code wait cap 200, paper 300; training critic LR 3e-4, paper 4e-4 | Record both code-faithful and paper-faithful configurations |
| Unsupported counterfactual labels | Predictor buffer update changes last acceleration to -3 and assigns label 0 without resimulation | Braking does not prove collision avoidance. Can confound action sensitivity figures and calibrated risk |
| Padding mismatch | Completed training sequences zero-padded; live inference repeats first state/action | Distribution mismatch for short histories |
| Risk inference uses training mode | train_sac predicts under no_grad without eval(); predictor has dropout | no_grad does not disable stochastic dropout |
| Risk optimizer recreated | train_model builds AdamW every call | Optimizer state is discarded; record this when matching baseline |
| Error swallowing | Predictor returns risk 0.5 on selected CUDA failures; training can skip batches | Require nonzero successful updates and fail clearly on OOM during research |
| Seeds incomplete | Only Torch seeded in train/test; Python and NumPy not seeded; SUMO hard-coded seed 1; Env.reset seed unused | CLI seed does not fully specify experiments |
| Reporting ambiguous | monitor counts colliding vehicle IDs and entries into junction directions; repeated IDs and multiple junction passages need deduplication | Vehicle involvements, unique crashes, unique vehicles, and junction crossings are different denominators |

**Assets present:** SUMO configuration, ~14 MB network, ~12 MB route file, view file, and old rawdump output. Both declared input XML files exist. Network XML version 1.9; network contains 4753 junction elements; selected junctions 229, 499, 332, 334 exist. Route file includes 30,502 vehicle elements and six flows; this is not a newly generated single ideal symmetric intersection. No explicit vType element was found: the actual underlying car-following model should be verified rather than inferred from Python objects named `IDM`.

**Not found in this checkout:** pretrained actor/critic/risk weights, FCFS implementation, Wang training/evaluation implementation or weights, batch evaluation/seed runner, ablation configurations, raw result tables, reproducible Table II aggregation script, environment lock, or LICENSE file. `rawdump.xml` is a simulation trace, not proof of trained performance or a model checkpoint. A commented alternative predictor import is not an active missing-file dependency.

**Protocol mismatch:** the supplied task config operates on four mapped junction IDs, whereas the final paper describes a symmetric four-way scenario. Clarify the precise subnetwork, lane layout, demand, IDM parameters and measurement windows that generated each table. Replacing the supplied map with a new toy crossroad creates a separate experiment.

## What executed in this session

Main isolated directory: `/workspace/research/.venv`; Python 3.10.22, Torch 1.13.1+cpu, NumPy 1.23.5, traci/sumolib 1.12.0, Gym 0.23.1, Gymnasium 0.26.3, Ray/RLlib 2.5.0, PyArrow 12.0.1, pandas 1.5.3, lxml 5.4.0, TensorBoard 2.13.0. Full resolved pins are in `requirements-smoke.lock.txt`. SUMO binary 1.12.0 is separately installed in `.sumo-venv` with Python 3.8.20 and `requirements-sumo.lock.txt`: the official wheel advertises a generic Python filename but its internal WHEEL tags are cp38. It ran under the initial 3.10 environment, but repeat setup's package compatibility check caught the tag mismatch. Isolating the unmodified official wheel under Python 3.8 resolved the compatibility check; no metadata/signature/TLS verification was disabled. This is a **compatibility environment**, not the original inconsistent requirements and not a validated GPU environment.

- Passed: imports of Env, TensorBoard and Torch; simulator reports SUMO 1.12.0.
- Passed: actual upstream risk predictor forward pass, BCE backward pass with finite gradients, eval-mode determinism, and one nonzero balanced-buffer training call, all on synthetic inputs. Predictor parameter count: 1,223,681. Synthetic loss values are not prediction-quality evidence.
- Passed: upstream Env reset and 100 control steps, 290 observations and 290 finite rewards. Policy was fixed zero acceleration. Risk reward was disabled. All four configured junctions and original input assets were used; only generated-output paths were redirected through a derived local configuration.
- Failed as expected from audit: original dependency resolution, evaluation argument parser and a deliberately short original training run at first checkpoint save (5 steps). These are genuine failures, not passing regression tests.
- The failed original script also terminated its SUMO connection abruptly and produced native shutdown diagnostics. Those diagnostics are not an additional diagnosed simulator defect. The successful smoke path closes its simulator in finally.
- Repeatability verified: after isolating the SUMO wheel, `install_smoke.sh` exited 0; both environments passed package compatibility checks and the 100-step smoke passed again (290 observations/rewards). Logs are in `runs/repeat-setup.log` and `runs/smoke-results.json`.
- Not run: corrected end-to-end training, learned-policy checkpoint round trip, full original experiment, FCFS/Wang evaluation, any GPU run, uncertainty extension, or novel-method comparison.

No full RL training or manuscript drafting was started. Code2 remained untouched. Generated smoke files are outside the author checkout. Python 3.10 bytecode accidentally created by the initial import probe was removed; existing tracked bytecode was preserved.

## Hardware and installation plan for Neo

**Recommended local platform:** Windows + VS Code Remote WSL + Ubuntu 22.04; headless SUMO first. This avoids Linux-only historical-package friction and GUI/display requirements. No physical vehicle experiment is needed for the proposed simulation study.

1. Archive/pin upstream commit and record SHA256 of network/route/config inputs. Preserve it as a read-only reference; implement necessary corrections in a separately labelled experimental implementation, with a change ledger.
2. Use Python 3.10 as an explicit compatibility deviation. SUMO binary, sumolib and traci should all be 1.12.0 for reference runs. The pip wheel of SUMO 1.12.0 executed successfully here; unversioned apt install does not pin SUMO.
3. Start with the supplied CPU lock and smoke script, including the separate SUMO tool environment described above. Put `.venv/bin` and then `.sumo-venv/bin` on PATH so Python uses 3.10 and the SUMO launcher uses its own 3.8 shebang. Local WSL paths must be adapted consistently from `/workspace/research`. uv is available on this cloud machine; install a trusted uv/Conda tool locally if needed. GPU environment should be a **separate environment**: Torch 1.13.1+cu117 is a candidate matching the authors' requested Torch generation, subject to the Windows NVIDIA driver/WSL CUDA support check. It has not been tested here. Avoid mixing CPU and GPU Torch locks.
4. Validate nvidia-smi in Windows and WSL; verify torch.cuda.is_available(), finite forward/backward on cuda and checkpoint reload on cuda/cpu. Measure peak allocated/reserved VRAM. Do not infer GPU readiness from driver presence alone.
5. Required corrected-baseline checks before training: [B,98] -> [B,1] policy; action/vehicle ordering; nonterminal Bellman bootstrap and true terminal masks; truncation treatment; one real-data SAC update; one risk update with both actual label classes; checkpoint save/load; seeded deterministic evaluation; exact simulator output/metric definitions. Use common scripted observations to compare action/metric mappings.
6. Run two or three short 1000-step episodes, not 500 episodes, and profile simulator time, risk update time, memory and successful gradient-update counts. Then estimate full budget from measured throughput. Larger training is conditional on correctness gates.

The 16 GB A4000 is the preferred training GPU; the 8 GB RTX 2060 SUPER is a plausible smoke/ablation device. This is an engineering estimate, not a measured runtime promise. The predictor is ~1.22 M parameters. Correct 1-D SAC replay at 1 million items occupies ~0.79 GB in raw state/action/reward/done tensors before other allocations, and upstream stores it on the chosen device. CPU replay is a possible engineering adaptation, applied identically to all methods and documented.

The dominant cost may be SUMO/Python and repeated predictor training, not VRAM. At full 20,000 risk-buffer items, upstream train_model can do floor(20000/128)*3 = 468 risk minibatches per call, invoked every 200 training-loop steps. A 500,000-step run can therefore become expensive. First measure this schedule; do not silently reduce its budget for only one method.

## Limits and relevance to uncertainty-aware decision-making

Useful foundation: state/action-aware risk guidance, low-dimensional simulation and continuous control. But SAC policy entropy is not epistemic confidence; sigmoid risk is not automatically a calibrated probability; outcome classification is not a Bayesian belief about intention. No explicit epistemic/aleatoric decomposition or guaranteed safety constraint is established.

Balanced outcome sampling shifts the predictor's training prevalence. For pure case-control selection, posterior odds can be corrected using deployment versus training priors, but synthetic safe labels and policy-dependent trajectories break that simple assumption. Calibrate with a held-out, naturally sampled set and report Brier/NLL, reliability and action-conditioned errors. A change from +3 to -3 in the predictor is not causal proof that braking avoids an accident: validate action effects with simulator rollouts from the same state.

Known route/lane movement, global occupancy, a mostly coordinated controlled population and simulator-native features limit the original uncertainty problem. Mixed human-driver style, delayed/ambiguous observed behaviour, and within-encounter style switching would be a **modified partially observed task**. No occluded-object detection or occlusion extension is proposed.

Wang's attached paper already studies mixed traffic, RV-percentage drops, V2V errors, unseen intersections and traffic-rule adaptation. Therefore “we add mixed traffic” or “we add behaviour variation” alone is not a justified novelty claim. Dresner–Stone's system is a reservation protocol with an intersection manager, not the same policy architecture or communication assumption. An FCFS adaptation is not automatically a full reproduction of that 66-page system.

## Candidate extension: quantum-inspired belief, with falsifiable purpose

**Hypothesis, not established novelty:** a context-sensitive matrix belief over latent yielding/assertive/committed behaviour modes may represent ambiguity and adapt to style switches better than simpler matched beliefs, improving calibrated action-conditioned tail-risk estimates and decisions.

Keep routes visible and lane-consistent in the first study; hide behaviour modes/parameters only. Observe positions, speeds and realised accelerations of fully visible vehicles. Generate physically specified variation in time headway, desired speed, reaction delay and yielding policy, plus controlled switches. Mode labels are available only to the evaluation oracle, never to deployed policies. If the original aggregate 98-D state loses essential identity/kinematic history, augment observations **for every method in the modified-task track**.

For K=3 modes, use a small complex Hermitian positive-semidefinite density matrix:

`rho_t = L_t L_t† / tr(L_t L_t†)`.

Diagonal entries can encode mode probabilities in a stated basis. Off-diagonal terms are learned contextual features, not evidence of physical quantum states in drivers. A valid contextual update may use Kraus operators `rho' = sum_m K_m rho K_m†`, with `sum_m K_m† K_m = I`, or a justified normalized observation-conditioning map. Enforce PSD, unit trace and numerical stability. Any unitary/context transform and evidence-update ordering must be defined before experiments, rather than chosen after seeing test results.

A calibrated event effect `0 <= E_a <= I` gives `p(event|history,a) = tr(rho E_a)`. If every effect and update is diagonal in one common basis, coherence has no observable role: the model reduces to a classical mixture. Do not claim an advantage from phases that the prediction path cannot use. Conversely, complex networks/density matrices have classical real-valued implementations; the research claim can only be a useful inductive bias under a particular budget, not quantum computational superiority.

Use a common action-conditioned rollout/cost model to generate a distribution of horizon costs for every belief variant. Risk-sensitive selection can minimize expected efficiency/comfort cost plus `lambda * CVaR_0.95(C)` and a common calibrated collision-probability constraint. CVaR is applied to a defined trajectory cost with near-miss/comfort/delay components and a dominant crash term; collision, intervention and efficiency metrics are still reported separately. CVaR on a binary crash indicator alone often saturates and provides poor discrimination. Predictive uncertainty should separate within-model outcome variation from ensemble disagreement, while acknowledging this decomposition depends on the model.

The original risk model is test-time disabled. An online belief/risk selector is therefore an architectural change. Include a control that gives the classical model the exact same online selector, prediction horizon, candidate accelerations and intervention budget. Otherwise the experiment tests online risk control versus no online risk control, not quantum-inspired belief.

## Matched controls and experimental footing

Use one common observation encoder, data, train/validation/test partitions, action set or continuous actor, world model, reward/cost, calibration split, simulator safety settings, evaluation seeds and tuning budget. Match parameter counts and training FLOPs as closely as practicable, and also report measured latency. Separate training transitions, agent transitions and gradient updates because the number of controlled vehicles varies.

| Variant | Belief/history representation | Decision rule | What comparison isolates |
|---|---|---|---|
| Original SAC-RWB | Original history predictor, test-time actor only | Original actor | Original-task reference, after documented corrections |
| Classical Bayes/HMM/IMM | Categorical mode posterior with transition/change model | Common risk-sensitive selector | Interpretable uncertainty/control baseline |
| GRU/Transformer | Real-valued recurrent/context representation | Same selector | Generic memory/nonlinearity explanation |
| Deep ensemble | Calibrated classical posterior/predictor, matched total budget | Same selector | Epistemic uncertainty benefit |
| Quantum-inspired diagonal | rho constrained diagonal | Same selector | Matrix bookkeeping without coherence |
| Quantum-inspired full | Valid rho and context/effect maps | Same selector | Candidate coherence/context inductive bias |
| Real matrix/feature control | Same degrees of freedom as full complex representation, expressed in real coordinates | Same selector | Extra capacity/complex arithmetic explanation |
| Oracle mode | True current mode/change state | Same selector | Diagnostic ceiling; privileged information, not fair deployable competitor |

Factorial ablation: belief representation × expected-cost/CVaR decision rule. Compare diagonalized/dephased inference and retrained diagonal model separately; a test-only ablation can create distribution shift. Include no-bias attention and risk-without-uncertainty controls where useful, without changing the upstream baseline and calling it identical.

**Two experiment tracks:** A reproduces the original task as closely as evidence allows; B evaluates the explicitly modified uncertain-behaviour task. In B, rerun the corrected baseline and all controls under the same modifications, and label any shared added information/runtime risk component. Do not compare proposed-method results from B against published numbers from A. Do not give the proposed method extra sensing, hidden modes, a larger action set, privileged future trajectories or a separate safety shield.

Report actual crash events and vehicle involvements separately; stopped-in-intersection proxy; near misses/TTC or PET under an explicit conflict-point definition; completed trips, entry flow and unfinished vehicles; waiting/delay, queue, jerk and intervention rates; calibration, adaptation delay and mean/tail inference latency. Declare warm-up, duration, right-turn handling, demand and seeds. Aggregate with confidence intervals clustered by independent episode/training seed; overlapping history windows and correlated vehicles are not independent trials. A zero-crash run is not a safety guarantee: with independent trials, zero in N gives an approximate 95% upper bound 3/N, and dependent events need appropriate clustered analysis.

## Early go/no-go before full RL or writing

**Gate 0 — baseline:** fix and verify the documented structural faults in a separately labelled implementation; confirm paper/code discrepancies or record unavoidable assumptions. Need save/load, finite real-data updates, nonterminal bootstrap, correct scalar actions, clean rollout termination, all event definitions and a deterministic seeded replay. Smoke success alone does not pass this gate.

**Gate 1 — uncertainty is observable and consequential:** avoid training a new SAC policy. Start with fixed, identical ego policies and a small candidate acceleration set, e.g. {-3,-1.5,0,1.5,3}. From independent SUMO saved states, perform matched counterfactual rollouts with reproducible random-state restoration, identical observations and randomized latent behavioural modes. No test-time oracle access. History excludes future outcomes. Pilot roughly 2,000–5,000 independent encounters across separate generator seeds; decide the final sample size from observed event prevalence and a prospective power calculation, not from a convenient number of windows. Split by complete encounter, generator seed and behaviour regime, not randomly by timestep. Include familiar modes, held-out mode parameters, abrupt switches and gradual drift. If risk never varies across plausible actions or the observations make modes perfectly obvious, redesign the experiment or reject this extension.

Train only the small belief/risk heads on this common dataset; frozen shared features and equal compute. Include persistent-HMM/change-point, GRU, ensemble, diagonal/full matrix and capacity-matched real controls. Fit calibration only on validation data. Use at least three optimisation seeds for screening, then a fresh held-out confirmation set and preferably five or more seeds if the screen passes. Report screening uncertainty rather than treating three seeds as definitive evidence.

**Gate 2 — pre-register a practical threshold before test inspection.** Example proposed criteria, not empirical claims: at least 5% relative Brier improvement versus the strongest tuned classical control, paired/clustered 95% interval excluding zero on an independently confirmed switch/drift evaluation; no material degradation of familiar-regime calibration; and a closed-loop tail-cost improvement using the same frozen selector, with waiting-time noninferiority margin 5% and no increase in measured crashes/intervention burden. Define low-prevalence event budgets/power first; if confidence intervals are too wide, result is inconclusive rather than success. An illustrative 20 ms p95 decision latency budget must be measured on Neo's target machine and adjusted to the actual decision rate before preregistration.

**Go:** full matrix reliably outperforms diagonal, strong HMM/GRU/ensemble and the matched real-valued control at a useful compute budget; calibration/adaptation improvement translates into decisions; no information leakage, cherry-picked seeds or safety-rule differences explain it.

**No-go for the quantum component:** only the risk-sensitive selector helps; phases/coherence add no stable benefit; a matched classical model ties/wins; improvements require extra parameters/information; outcomes cannot be calibrated from genuine rollouts; or experimental uncertainty remains too large. Keep a successful classical uncertainty-aware framework if warranted, but it still needs a separate literature/novelty assessment. Do not force a quantum label.

These numerical margins are draft design choices to discuss with Neo, not accepted universal journal standards. Broader prior-art verification on quantum-like cognition/belief models, classical POMDP/change-point models, uncertainty-aware intersection planning and distributional/CVaR RL remains necessary before any novelty claim. The base paper's own refs [19] (TVT 2025) and [20] (ITSC 2025) are immediate relevant leads, not papers whose results were independently verified here.

## Saved cloud configuration

The cloud configuration draft successfully saved complete `install_script` and `start_skill` instructions for this CPU component-smoke workflow. Other fields were not changed. The platform returned `status: saved` and `requires_publish: true`. Neo can review/save the changes in environment settings and publish the environment to retain the prepared snapshot. Saving is not publication, new-task restoration has not been verified, and neither the saved instructions nor current component smoke establish corrected-baseline/full-paper readiness.

## Author clarification list

Request the final-paper code/config revision; exact map/subnetwork/lane and traffic settings for each table; collision/proxy event counting and denominators; 1500 rounds versus steps and warm-up; trained weights and baseline/ablation runners; scalar-action and terminal-mask intent; negative/positive risk sign and efficiency rank/slicing; rationale/validation for synthetic braking labels; physical simulator car-following settings; and the training/test seed lists. Reimplementation can proceed transparently on documented assumptions, but exact numerical reproduction remains qualified until those ambiguities are resolved.
